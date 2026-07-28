"""guard_node/permit_phase_directive 시나리오 테스트.

LLM을 호출하지 않는다 - 둘 다 상태(AgentState)만 보고 결정론적으로 동작하는
함수라, 실제 대화 턴을 messages 시퀀스로 흉내내는 것만으로 정확히 재현
가능하다. 2026-07-28 세션에서 실사용 중 발견한 버그(disclosed_stage 래칫
때문에 되짚은 뒤 단계를 건너뛰는 문제, 세션 f377454d)를 회귀 테스트로
남긴다.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent import _STAGE_DOMAIN, guard_node, permit_phase_directive

# permit_synthesis가 채워져 있어야 [Step 1-2]/[Step 1-3] 언급이 synthesis-gap
# 위반(guard_node 위반 3번)에 안 걸린다 - 이 파일의 테스트는 disclosed_stage
# 갱신 로직 자체를 보는 것이라, 관련 없는 위반까지 같이 걸리면 원인 파악이
# 헷갈린다.
_SYNTHESIS = {"required_documents": ["건축ᆞ대지 현황도"], "pre_diagnosis_items": ["주차대수"]}


def _base_state() -> dict:
    return {
        "messages": [],
        "case_facts": {"_classified": True},
        "disclosed_stage": {},
        "permit_synthesis": dict(_SYNTHESIS),
    }


def _apply(state: dict, update: dict) -> dict:
    """guard_node가 반환하는 부분 업데이트를 그래프의 기본 덮어쓰기 리듀서처럼
    병합한다(messages만 append, 나머지는 overwrite)."""
    new_state = dict(state)
    for key, value in update.items():
        if key == "messages":
            new_state["messages"] = state.get("messages", []) + value
        else:
            new_state[key] = value
    return new_state


def _disclosed(state: dict) -> int:
    return state.get("disclosed_stage", {}).get(_STAGE_DOMAIN, 0)


def test_permit_phase_directive_none_before_classification():
    state = {"case_facts": {}}
    assert permit_phase_directive(state) is None


def test_sequential_walkthrough_then_construction_transition():
    """Step1-1→1-2→1-3 순서로 진행하고, 다 끝나면 Step2(공사) 전환 넛지가
    나오고, get_construction_guide 호출 후엔 더 이상 넛지가 안 나온다."""
    state = _base_state()

    assert "[Step 1-1]까지만" in permit_phase_directive(state)

    state["messages"].append(AIMessage(content="[Step 1-1: 상황 분석+필요 절차] 신고 대상입니다."))
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 1
    assert "[Step 1-2]까지만" in permit_phase_directive(state)

    state["messages"].append(AIMessage(content="[Step 1-2: 필수 서류] 건축ᆞ대지 현황도"))
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 2
    assert "[Step 1-3]까지만" in permit_phase_directive(state)

    state["messages"].append(
        AIMessage(content="[Step 1-3: 사전 진단] 주차대수 확인 필요. 예상 소요 기간은 신고 기준 3~5일입니다.")
    )
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 3

    directive = permit_phase_directive(state)
    assert "Step 2(공사) 안내를 시작하지 않았습니다" in directive

    call_id = "c1"
    state["messages"].append(
        AIMessage(content="", tool_calls=[{"name": "get_construction_guide", "args": {}, "id": call_id}])
    )
    state["messages"].append(
        ToolMessage(content="[착공ᆞ공사 단계 안내]...", tool_call_id=call_id, name="get_construction_guide")
    )
    state["messages"].append(AIMessage(content="다음은 Step 2(공사) 단계입니다. 착공신고부터 안내드릴게요."))
    state = _apply(state, guard_node(state))

    assert state["disclosed_stage"].get("construction_guide") == 1
    assert permit_phase_directive(state) == "[진행 상태] Step 1~2(건축 인허가ᆞ공사) 안내가 모두 끝났습니다."


def test_revisit_lowers_stage_instead_of_skipping():
    """회귀 테스트(세션 f377454d) - Step1-3까지 간 뒤 Step1-2를 다시 물으면
    disclosed_stage가 2로 내려가고, 이어가면 Step1-4로 건너뛰지 않고
    Step1-3부터 다시 진행된다."""
    state = _base_state()
    state["disclosed_stage"] = {"case_facts": 3}
    state["messages"] = [
        HumanMessage(content="서류 다시 알려줘"),
        AIMessage(content="[Step 1-2: 필수 서류] 다시 설명드릴게요."),
    ]

    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 2, "되짚으면 disclosed_stage가 내려가야 한다"
    assert "[Step 1-3]까지만" in permit_phase_directive(state), "다음 턴은 1-3부터지 1-4로 건너뛰면 안 된다"

    state["messages"] += [
        HumanMessage(content="계속"),
        AIMessage(content="[Step 1-3: 사전 진단] 다시 안내드릴게요."),
    ]
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 3


def test_step1_4_marker_no_longer_recognized():
    """2026-07-28에 Step1을 4단계에서 3단계로 줄였다 - 옛 [Step 1-4] 헤더가
    실수로 다시 등장해도 더 이상 마커로 인식되면 안 된다."""
    state = _base_state()
    state["disclosed_stage"] = {"case_facts": 3}
    state["messages"] = [
        HumanMessage(content="더 자세히"),
        AIMessage(content="[Step 1-4: 예상 소요 기간] 3~5일 정도입니다."),
    ]
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 3


def test_unrelated_answer_does_not_touch_disclosed_stage():
    """[Step 1-N] 마커가 아예 없는(다른 도메인) 답변은 disclosed_stage를
    건드리지 않는다."""
    state = _base_state()
    state["disclosed_stage"] = {"case_facts": 3}
    state["messages"] = [
        HumanMessage(content="간판은 신고 대상인가요?"),
        AIMessage(content="네, 벽면이용간판은 신고 대상입니다."),
    ]
    update = guard_node(state)
    assert "disclosed_stage" not in update or _apply(state, update)["disclosed_stage"].get(_STAGE_DOMAIN) == 3


def test_synthesis_gap_still_triggers_retry():
    """record_permit_synthesis 기록 없이 [Step 1-2]를 언급하면 여전히
    재시도를 유도해야 한다(회귀 없음 확인)."""
    state = _base_state()
    state["permit_synthesis"] = {}
    state["messages"] = [
        HumanMessage(content="서류 알려줘"),
        AIMessage(content="[Step 1-2: 필수 서류] 건축ᆞ대지 현황도가 필요합니다."),
    ]
    update = guard_node(state)
    assert any(
        isinstance(m, ToolMessage) and m.name == "_answer_guard"
        for m in update.get("messages", [])
    )
