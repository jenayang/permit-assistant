"""_forced_record_tool(도메인별 record_* 강제 호출) 회귀 테스트.

LLM을 호출하지 않는다 - 상태와 최근 발화만 보고 "이번 턴에 어떤 도구를
tool_choice로 강제할지"를 결정론적으로 고르는 함수라, 상태를 흉내내는 것만으로
정확히 검증된다.

배경(2026-08-02 실측): 운영과 동일한 15개 도구를 노출한 상태에서 record_* 호출률이
0/3이었고(3회 반복 전부 동일 = 결정론적), tool_choice로 강제하면 3/3이었다.
프롬프트 강화로는 못 고치고 API 레벨 강제로만 고쳐지는 문제라, 강제가 (a) 필요한
자리에서 확실히 걸리고 (b) 불필요한 자리로 번지지 않는지를 양방향으로 건다.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent import _forced_record_tool


def _state(*messages, **facts) -> dict:
    return {"messages": list(messages), **facts}


def test_forces_record_when_domain_raised_first_time():
    """유저가 처음 꺼낸 도메인은 그 도메인의 record_* 를 강제해야 한다 -
    이게 없으면 15개 도구 환경에서 실측 0% 호출률로 돌아간다."""
    state = _state(HumanMessage(content="가게 앞에 벽면에 붙이는 간판 달려고요"))
    assert _forced_record_tool(state) == "record_signage_facts"


def test_no_force_once_domain_facts_exist():
    """이미 그 도메인 사실이 기록돼 있으면 강제하지 않는다 - 강제는 도메인당
    사실상 첫 기록 한 번뿐이어야 매 턴 낭비되지 않는다."""
    state = _state(
        HumanMessage(content="간판 얘기 더 해주세요"),
        signage_facts={"sign_type": "벽면이용간판"},
    )
    assert _forced_record_tool(state) is None


def test_no_force_twice_in_same_turn():
    """이번 턴에 이미 그 도구가 불렸으면 또 강제하지 않는다. 모델이 강제된
    호출에 전부 None을 채워 보내면 facts가 여전히 비어 있어, 이 방어가 없으면
    같은 턴에서 무한히 다시 강제된다."""
    state = _state(
        HumanMessage(content="간판 달려고요"),
        AIMessage(content="", tool_calls=[{"name": "record_signage_facts", "args": {}, "id": "t1"}]),
        ToolMessage(content="기록됨.", tool_call_id="t1", name="record_signage_facts"),
    )
    assert _forced_record_tool(state) is None


def test_no_force_for_definition_question():
    """정의를 묻는 질문은 기록할 사실이 없으므로 강제하지 않는다(강제를 거는
    것보다 안 거는 쪽이 안전 - 기존 동작으로 돌아갈 뿐이다)."""
    for text in ("건축법이 뭐야?", "연면적의 정의가 뭔가요?", "건폐율과 용적률 차이가 뭐죠?"):
        assert _forced_record_tool(_state(HumanMessage(content=text))) is None, text


def test_no_force_when_no_domain_keyword():
    """도메인 키워드가 없는 발언에는 강제하지 않는다."""
    assert _forced_record_tool(_state(HumanMessage(content="감사합니다"))) is None


def test_no_force_when_last_message_is_not_human():
    """도구 실행 직후처럼 유저 턴의 시작이 아니면 판단 대상이 아니다."""
    state = _state(
        HumanMessage(content="간판 달려고요"),
        AIMessage(content="어떤 종류인가요?"),
    )
    assert _forced_record_tool(state) is None


def test_permit_domain_is_forced_too():
    """건축 도메인도 같은 규칙으로 강제된다(도메인별 특례 없음)."""
    state = _state(HumanMessage(content="연면적 100제곱미터 사무실을 카페로 용도변경하려고요"))
    # DOMAIN_CONFIGS 순서상 permit이 먼저 걸린다 - 한 턴에 하나만 강제하고
    # 나머지 도메인은 사실이 쌓이면서 다음 턴에 자연스럽게 처리된다.
    assert _forced_record_tool(state) == "record_case_facts"
