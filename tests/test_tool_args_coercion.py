"""LLM이 보낸 raw tool args의 타입 강제 변환 회귀 테스트.

배경(2026-08-02 실측): 폴백 모델(Cerebras gemma-4-31b)은 정수 필드를 문자열로
직렬화해 보낸다. 값 자체는 정확한데(사무실→8, 카페→7 매핑까지 전부 맞음) 타입만
다르다. 그게 그대로 상태에 들어가면 classify_case에서 `facts["floors"] < 3`이
TypeError로 크래시한다. ToolNode는 도구를 실행할 때 pydantic 검증을 거치지만
_agent_result는 tc["args"]를 날것으로 읽어 상태에 넣기 때문에 그 변환을
우회하고 있었다.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agent import _agent_result, _coerce_tool_args
from src.agents.permit import classify_case


def _ai(tool: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": "t1"}])


def test_int_fields_coerced_from_string():
    """폴백 모델이 보내는 문자열 정수를 int로 바꿔야 한다."""
    got = _coerce_tool_args(
        "record_case_facts",
        {"floors": "3", "current_facility_group": "8", "desired_facility_group": "7"},
    )
    assert got == {"floors": 3, "current_facility_group": 8, "desired_facility_group": 7}
    assert all(isinstance(v, int) for v in got.values())


def test_coercion_keeps_only_sent_keys():
    """model_dump()가 안 보낸 필드까지 None으로 채워 돌려주므로, 보낸 키만
    남겨야 한다 - 안 그러면 상태에 불필요한 None이 잔뜩 섞인다."""
    got = _coerce_tool_args("record_case_facts", {"act_type": "신축"})
    assert got == {"act_type": "신축"}


def test_invalid_args_fall_back_to_original():
    """검증 실패 시 예외를 던지지 말고 원본을 그대로 써야 한다 - 여기서
    터지면 대화 자체가 끊긴다."""
    bad = {"floors": "세 개"}
    assert _coerce_tool_args("record_case_facts", bad) == bad


def test_unknown_tool_passes_through():
    assert _coerce_tool_args("존재하지_않는_도구", {"a": 1}) == {"a": 1}


def test_agent_result_applies_coercion_to_state():
    """_agent_result가 상태에 넣는 값이 이미 변환된 상태여야 한다 -
    이 경로가 실제로 규칙 엔진에 들어가는 값이다."""
    update = _agent_result(_ai("record_case_facts", {"floors": "2", "act_type": "신축"}))
    assert update["case_facts"] == {"floors": 2, "act_type": "신축"}


def test_rule_engine_no_longer_crashes_on_fallback_model_output():
    """회귀의 핵심 - 폴백 모델이 보낸 그대로의 args가 규칙 엔진까지 흘렀을 때
    TypeError로 죽지 않고 정상 판정돼야 한다."""
    raw = {"act_type": "신축", "size_sqm": 150, "floors": "3", "land_zone": "관리지역"}
    facts = _agent_result(_ai("record_case_facts", raw))["case_facts"]
    assert classify_case(facts) == "건축허가"


def test_float_and_bool_fields_also_coerced():
    """정수 외 타입도 같은 경로로 정리된다(도메인마다 필드 타입이 다름)."""
    got = _coerce_tool_args("record_fire_facts", {"size_sqm": "400"})
    assert got == {"size_sqm": 400.0}
    got = _coerce_tool_args("record_signage_facts", {"sign_type": "돌출간판", "floor": "3"})
    assert got == {"sign_type": "돌출간판", "floor": 3}


def test_lookup_tools_persist_address_into_case_facts():
    """lookup_land_zone/lookup_building_ledger 호출 시 address가 case_facts에도
    저장돼야 한다 - record_case_facts엔 address 필드가 없어서(법령 판정에
    안 쓰임) 예전엔 이 값이 어디에도 안 남았다. 로드맵 Step1의 "사업 예정지
    주소 확인" substep(src/roadmap_model.py)이 case_facts.get("address")를
    보는데 아무도 안 채워서 영원히 미완료로 남는 회귀였다(2026-08-05,
    세션 b846cc22 - 주소를 입력하고 건축물대장까지 조회했는데도 체크박스가
    막혀있고 로드맵이 Step1에서 안 넘어감)."""
    update = _agent_result(_ai("lookup_building_ledger", {"address": "서울시 강남구 테헤란로 1"}))
    assert update["case_facts"] == {"address": "서울시 강남구 테헤란로 1"}

    update = _agent_result(_ai("lookup_land_zone", {"address": "서울시 마포구 월드컵로 2"}))
    assert update["case_facts"] == {"address": "서울시 마포구 월드컵로 2"}
