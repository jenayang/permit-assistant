"""build_system_prompt(조건부 프롬프트 조립) 회귀 테스트.

LLM을 호출하지 않는다 - 조립은 상태(AgentState)와 최근 유저 메시지만 보고
결정론적으로 동작한다. 이 테스트가 지키려는 것은 두 방향이다:

1. **끄면 안 되는 걸 껐는지**(치명적) - 그 도메인의 수집 규칙이 프롬프트에서
   사라지면 판정 자체가 조용히 실패한다. 상태에 살아있는 도메인ᆞ키워드가
   등장한 도메인ᆞ식품접객업일 때의 소방(선제 판정)이 각각 켜지는지 확인.
2. **켜도 되는 걸 안 켰는지**(비용) - 무관한 도메인 블록이 계속 붙어 있으면
   조립을 한 의미가 없다. 간판만 물어본 상태에서 건축ᆞ식품 블록이 빠지는지
   확인(2026-08-01 실측에서 "허가"ᆞ"신고" 같은 범용어가 permit 블록을 항상
   켜버리던 문제를 잡아낸 지점이라, 그 회귀를 막는 게 이 테스트의 핵심).
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.prompts import (
    _ANSWER_PERMIT_STAGES,
    _COLLECT_COMMON,
    _COLLECT_FIRE,
    _COLLECT_FOOD,
    _COLLECT_PERMIT,
    _COLLECT_PERMIT_GUARD,
    _COLLECT_PERMIT_LOOKUP,
    _COLLECT_PERMIT_NEW_BUILD,
    _COLLECT_PERMIT_USE_CHANGE,
    _COLLECT_PROGRESS,
    _COLLECT_SIGNAGE,
    _PROMPT_ACCURACY,
    _PROMPT_HEADER,
    _PROMPT_TOOLS,
    SYSTEM_PROMPT,
    build_system_prompt,
)

_DOMAIN_BLOCKS = {
    "permit": _COLLECT_PERMIT,
    "food": _COLLECT_FOOD,
    "fire": _COLLECT_FIRE,
    "signage": _COLLECT_SIGNAGE,
    "progress": _COLLECT_PROGRESS,
}


def _state(text: str = "", **kwargs) -> dict:
    return {"messages": [HumanMessage(content=text)] if text else [], **kwargs}


def _active(prompt: str) -> set[str]:
    """조립된 프롬프트에 실제로 포함된 도메인 블록 이름 집합."""
    return {name for name, block in _DOMAIN_BLOCKS.items() if block in prompt}


def test_always_on_blocks_present_in_every_assembly():
    """도메인과 무관한 블록(역할 소개ᆞ도구 선택ᆞ정확성 원칙)은 어떤 상태에서도
    빠지면 안 된다 - 근거 인용 규칙이 빠지면 guard_node의 인용 검증과 어긋난다."""
    for state in (_state(), _state("간판 달고 싶어요"), _state("아무말")):
        prompt = build_system_prompt(state)
        assert _PROMPT_HEADER in prompt
        assert _PROMPT_TOOLS in prompt
        assert _PROMPT_ACCURACY in prompt


def test_eager_recording_principle_survives_without_permit_block():
    """회귀 테스트(2026-08-01 스모크 테스트에서 발견) - "이번 턴에 바로
    기록하라"는 원칙은 원래 permit 블록 안에 묻혀 있어서, 간판 전용 대화처럼
    permit이 꺼지는 턴에는 그 지시까지 통째로 사라졌다. 그 결과 사용자가
    "벽면에 붙이는 간판"이라고 sign_type을 이미 말했는데도
    record_signage_facts가 한 번도 안 불렸다. 도메인 무관한 공통 원칙이므로
    permit 여부와 상관없이 항상 붙어 있어야 한다."""
    state = _state("간판 신고해야 하나요?", signage_facts={"sign_type": "벽면이용간판"})
    prompt = build_system_prompt(state)
    assert _COLLECT_COMMON in prompt
    assert _COLLECT_PERMIT not in prompt  # permit은 실제로 꺼진 상태여야 유효한 검증


def test_full_prompt_contains_every_block():
    """SYSTEM_PROMPT(전체 조립본)에는 모든 블록이 들어있어야 한다 - 블록을
    새로 추가하고 전체 조립본에 넣는 걸 잊으면 여기서 걸린다."""
    for block in _DOMAIN_BLOCKS.values():
        assert block in SYSTEM_PROMPT
    for block in (
        _COLLECT_PERMIT_GUARD, _COLLECT_PERMIT_NEW_BUILD,
        _COLLECT_PERMIT_USE_CHANGE, _COLLECT_PERMIT_LOOKUP,
    ):
        assert block in SYSTEM_PROMPT
    assert _ANSWER_PERMIT_STAGES in SYSTEM_PROMPT


def test_bootstrap_turns_on_permit_when_nothing_active():
    """대화 초반(아무 도메인도 안 잡힘)엔 건축 블록이 켜져야 한다 - 이 챗봇의
    진입점이 행위 유형 파악이라, 없으면 record_case_facts를 시작 못 한다."""
    assert "permit" in _active(build_system_prompt(_state()))
    assert "permit" in _active(build_system_prompt(_state("뭐부터 해야 하나요?")))


def test_state_keeps_domain_on_without_keyword():
    """이미 상태에 살아있는 도메인은 이번 메시지에 키워드가 없어도 켜져 있어야
    한다 - 턴마다 깜빡이면 그 사이 턴에 판정이 끊긴다."""
    active = _active(build_system_prompt(_state("응 알겠어", food_result="제과점영업")))
    assert "food" in active


def test_keyword_turns_on_domain_at_first_mention():
    """상태엔 아직 없지만 유저가 처음 꺼낸 도메인은 키워드로 켜져야 한다."""
    assert "signage" in _active(build_system_prompt(_state("간판도 달려고요")))
    assert "food" in _active(build_system_prompt(_state("카페 하려고요")))


def test_keyword_activation_sticks_even_if_tool_call_was_skipped():
    """회귀 테스트(2026-08-01) - 키워드로 켜진 도메인은 그 턴에 record_* 도구가
    안 불려 상태에 아무것도 안 남았더라도 이후 턴에서 계속 켜져 있어야 한다.

    최근 한 턴만 스캔하면: "카페 창업하려고요"(food 켜짐) → LLM이
    record_food_facts 스킵(실측으로 확인된 문제) → 다음 턴 "그럼 건폐율은?"에
    키워드도 상태도 없어 food 블록이 꺼지고, 수집 규칙이 사라졌으니 이후로도
    영영 안 불려 식품 판정 자체가 조용히 실패한다."""
    state = {
        "messages": [
            HumanMessage(content="카페 창업하려고요"),
            AIMessage(content="어떤 건물인가요?"),  # 도구 호출 없음 = 스킵된 턴
            HumanMessage(content="그럼 건폐율은 어떻게 되나요?"),
        ]
    }
    assert "food" in _active(build_system_prompt(state))


def test_domain_stays_on_after_unrelated_followup():
    """도메인이 켜진 뒤 무관한 발언이 이어져도 유지돼야 한다 - 블록이 턴마다
    깜빡이면 대화 흐름이 불안정해진다."""
    state = {
        "messages": [
            HumanMessage(content="간판 달려고요"),
            AIMessage(content="어떤 종류인가요?"),
            HumanMessage(content="일단 그건 나중에 정할게요"),
        ]
    }
    assert "signage" in _active(build_system_prompt(state))


def test_fire_turns_on_proactively_for_food_business():
    """식품접객업(food_result 있음)이면 유저가 소방을 안 물어봐도 소방 블록이
    켜져야 한다 - "식품접객업이면 소방도 함께 판정하라"는 선제 규칙을 잃으면
    guard_node의 소방 미판정 체크에 계속 걸린다."""
    prompt = build_system_prompt(_state("영업신고 알려줘", food_result="휴게음식점"))
    assert "fire" in _active(prompt)


def test_signage_only_conversation_drops_unrelated_blocks():
    """간판만 다루는 대화에선 건축ᆞ식품 블록이 빠져야 한다. "허가"ᆞ"신고" 같은
    범용어를 permit 키워드에 두면 이 절감이 통째로 사라지므로(2026-08-01 실측)
    그 회귀를 막는 테스트다."""
    state = _state("간판 신고해야 하나요?", signage_facts={"sign_type": "벽면이용간판"})
    active = _active(build_system_prompt(state))
    assert "signage" in active
    assert "permit" not in active
    assert "food" not in active


def test_act_type_specific_guardrails_are_gated():
    """행위 유형별 가드레일은 그 유형일 때만 붙어야 한다 - 85㎡ 규칙은
    증축ᆞ개축ᆞ재축(+신축 오적용 방지)에만, 용도변경 3필드ᆞ시설군 방향
    규칙은 용도변경에만 의미가 있다."""
    msg = "성수동에 가게 하려고요"

    new_build = build_system_prompt(_state(msg, case_facts={"act_type": "신축"}))
    assert _COLLECT_PERMIT_NEW_BUILD in new_build
    assert _COLLECT_PERMIT_USE_CHANGE not in new_build

    use_change = build_system_prompt(_state(msg, case_facts={"act_type": "용도변경"}))
    assert _COLLECT_PERMIT_USE_CHANGE in use_change
    assert _COLLECT_PERMIT_NEW_BUILD not in use_change

    # 둘 다 무관한 유형에서는 양쪽 다 빠진다.
    renovation = build_system_prompt(_state(msg, case_facts={"act_type": "대수선"}))
    assert _COLLECT_PERMIT_NEW_BUILD not in renovation
    assert _COLLECT_PERMIT_USE_CHANGE not in renovation


def test_unknown_act_type_keeps_both_guardrails():
    """act_type이 아직 없으면 무엇이 될지 모르므로 양쪽 가드레일을 다 붙여야
    한다 - 여기서 빼면 그 턴에 바로 85㎡ 오적용이나 용도변경 3필드 누락이
    나올 수 있다(켜두는 쪽이 안전)."""
    prompt = build_system_prompt(_state("성수동에 가게 하려고요", case_facts={}))
    assert _COLLECT_PERMIT_NEW_BUILD in prompt
    assert _COLLECT_PERMIT_USE_CHANGE in prompt


def test_land_zone_lookup_block_drops_once_known():
    """용도지역을 이미 확보했으면 자동조회 전략 블록은 뺀다."""
    msg = "성수동에 가게 하려고요"
    assert _COLLECT_PERMIT_LOOKUP in build_system_prompt(_state(msg, case_facts={}))
    assert _COLLECT_PERMIT_LOOKUP not in build_system_prompt(
        _state(msg, case_facts={"land_zone": "기타"})
    )


def test_permit_core_blocks_always_accompany_permit():
    """수집ᆞ판정금지 가드레일은 행위 유형과 무관하게 건축 블록과 함께 붙어야
    한다 - 판정금지가 빠지면 LLM이 허가/신고를 직접 계산하는 걸 막을 수 없다."""
    for act in (None, "신축", "용도변경", "대수선", "가설건축물"):
        facts = {} if act is None else {"act_type": act}
        prompt = build_system_prompt(_state("성수동에 가게 하려고요", case_facts=facts))
        assert _COLLECT_PERMIT in prompt
        assert _COLLECT_PERMIT_GUARD in prompt


def test_permit_stages_block_follows_permit_block():
    """[Step 1-N] 단계 규칙은 건축 블록과 함께 켜지고 함께 꺼져야 한다 -
    한쪽만 남으면 다른 도메인 답변에 [Step 1-N] 마커가 섞여 permit의 단계
    카운터가 오염된다(disclosed_stage 오염은 실제로 겪은 버그)."""
    permit_on = build_system_prompt(_state("용도변경 하려고요"))
    assert _ANSWER_PERMIT_STAGES in permit_on

    signage_only = build_system_prompt(
        _state("간판 문의요", signage_facts={"sign_type": "입간판"})
    )
    assert _ANSWER_PERMIT_STAGES not in signage_only


def test_assembly_never_exceeds_full_prompt():
    """어떤 상태에서도 조립 결과가 전체 조립본보다 길 수 없다(블록 중복 조립
    방지)."""
    states = [
        _state(),
        _state("카페 창업", case_facts={"act_type": "용도변경"}, food_result="제과점영업"),
        _state("전부 알려줘", case_facts={"_classified": True}, food_result="휴게음식점",
               fire_result=["소화기구"], signage_result="신고", task_progress={"construction": True}),
    ]
    for state in states:
        assert len(build_system_prompt(state)) <= len(SYSTEM_PROMPT)
