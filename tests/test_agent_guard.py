"""guard_node/permit_phase_directive 시나리오 테스트.

LLM을 호출하지 않는다 - 둘 다 상태(AgentState)만 보고 결정론적으로 동작하는
함수라, 실제 대화 턴을 messages 시퀀스로 흉내내는 것만으로 정확히 재현
가능하다. 2026-07-28 세션에서 실사용 중 발견한 버그(disclosed_stage 래칫
때문에 되짚은 뒤 단계를 건너뛰는 문제, 세션 f377454d)를 회귀 테스트로
남긴다.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.classify import _derive_ledger_facility_group
from src.guard import guard_node
from src.roadmap import (
    _STAGE_DOMAIN,
    _should_force_construction_guide,
    permit_phase_directive,
)


def _ledger_msg(content: str) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="x", name="lookup_building_ledger")


def test_derive_ledger_facility_group_maps_use_name():
    """건축물대장 조회 성공 + current_facility_group 미기록이면 주용도를
    시설군으로 자동 변환한다(guard 재시도 없이 코드가 채움)."""
    st = {"messages": [HumanMessage(content="q"), _ledger_msg("조회 결과:\n- 주용도: 제1종근린생활시설")],
          "case_facts": {}}
    assert _derive_ledger_facility_group(st) == 7


def test_derive_ledger_skips_when_already_recorded():
    st = {"messages": [HumanMessage(content="q"), _ledger_msg("- 주용도: 업무시설")],
          "case_facts": {"current_facility_group": 8}}
    assert _derive_ledger_facility_group(st) is None


def test_derive_ledger_none_when_lookup_failed():
    st = {"messages": [HumanMessage(content="q"), _ledger_msg("등록된 건축물대장이 없습니다")],
          "case_facts": {}}
    assert _derive_ledger_facility_group(st) is None


def test_derive_ledger_none_when_no_lookup():
    st = {"messages": [HumanMessage(content="q")], "case_facts": {}}
    assert _derive_ledger_facility_group(st) is None

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
    """Step1-1→1-2→1-3→1-4 순서로 진행하고, 다 끝나면 Step2(공사) 전환 넛지가
    나오고, get_construction_guide 호출 후엔 더 이상 넛지가 안 나온다.

    2026-08-03: 1-2(사전검토)→1-3, 1-3(서류준비)→1-4 전환에도 완료 확인 게이트가
    생겼다 - 확인 전엔 다음 단계로 못 넘어가고, task_progress로 완료를 기록해야
    다음 단계 안내가 나온다(기존에 있던 1-4→Step2 게이트와 동일한 패턴)."""
    state = _base_state()

    assert "[Step 1-1]까지만" in permit_phase_directive(state)

    state["messages"].append(AIMessage(content="[Step 1-1: 대상 판정] 신고 대상입니다."))
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 1
    assert "[Step 1-2]까지만" in permit_phase_directive(state)

    state["messages"].append(AIMessage(content="[Step 1-2: 사전 검토] 주차대수 확인 필요"))
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 2

    directive = permit_phase_directive(state)
    assert "사전 검토 항목은 확인해 보셨어요" in directive, "사전검토 완료 확인 전엔 1-3으로 넘어가면 안 된다"
    state["task_progress"] = {"pre_diagnosis_checked": True}
    assert "[Step 1-3]까지만" in permit_phase_directive(state)

    state["messages"].append(AIMessage(content="[Step 1-3: 설계ᆞ서류 준비] 건축ᆞ대지 현황도"))
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 3

    directive = permit_phase_directive(state)
    assert "서류 준비는 다 되셨어요" in directive, "서류준비 완료 확인 전엔 1-4로 넘어가면 안 된다"
    state["task_progress"] = {**state["task_progress"], "documents_prepared": True}
    assert "[Step 1-4]까지만" in permit_phase_directive(state)

    state["messages"].append(
        AIMessage(content="[Step 1-4: 신청ᆞ접수] 관할 구청에 접수합니다. 예상 소요 기간은 신고 기준 3~5일입니다.")
    )
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 4

    directive = permit_phase_directive(state)
    assert "신청서 접수는 다 하셨어요" in directive, "신청ᆞ접수 확인 전엔 Step 2를 곧장 안내하면 안 된다"

    state["task_progress"] = {"application_submitted": True}
    directive = permit_phase_directive(state)
    assert "신청ᆞ접수도 확인됐습니다" in directive

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
    Step1-3부터 다시 진행된다. 이 테스트는 disclosed_stage 되짚기 로직만
    보는 것이라, 완료 확인 게이트(2026-08-03)가 끼어들지 않도록 미리
    완료된 것으로 둔다."""
    state = _base_state()
    state["disclosed_stage"] = {"case_facts": 3}
    state["task_progress"] = {"pre_diagnosis_checked": True, "documents_prepared": True}
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


def test_step1_5_marker_not_recognized():
    """2026-08-03에 Step1을 3단계에서 4단계로 다시 늘렸다(대상판정ᆞ사전검토ᆞ
    서류준비ᆞ신청접수) - 상한을 벗어난 [Step 1-5] 같은 헤더는 마커로 인식되면
    안 된다."""
    state = _base_state()
    state["disclosed_stage"] = {"case_facts": 4}
    state["messages"] = [
        HumanMessage(content="더 자세히"),
        AIMessage(content="[Step 1-5: 예상 소요 기간] 3~5일 정도입니다."),
    ]
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 4


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


def test_construction_guide_gap_triggers_retry():
    """회귀 테스트(세션 5db53ccb, 2026-07-29) - Step1이 다 끝난 뒤
    permit_phase_directive가 get_construction_guide 호출을 지시하는데도,
    그 지시를 무시하고 "Step 2(공사)" 내용을 도구 호출 없이 텍스트로만
    설명하면 재시도를 유도해야 한다(RAG 검색 없이 절차를 지어내는 걸 막음)."""
    state = _base_state()
    state["disclosed_stage"] = {_STAGE_DOMAIN: 4}
    state["task_progress"] = {"application_submitted": True}
    state["messages"] = [
        HumanMessage(content="공사는 어떻게 진행돼?"),
        AIMessage(content="다음은 Step 2(공사) 단계입니다. 착공신고를 먼저 진행하셔야 해요."),
    ]
    update = guard_node(state)
    guard_msgs = [m for m in update.get("messages", []) if isinstance(m, ToolMessage) and m.name == "_answer_guard"]
    assert guard_msgs, "get_construction_guide 미호출인데도 재시도가 유도되지 않음"
    assert "get_construction_guide" in guard_msgs[0].content


def test_construction_guide_gap_ignored_before_application_submitted():
    """2026-08-03 회귀 - Step1-4까지 다 끝났어도 신청ᆞ접수(application_submitted)가
    아직 확인 안 됐으면, permit_phase_directive가 실제로 지시하는 건 "Step 2
    안내"가 아니라 "접수 확인 질문"이라 이 위반 체크 자체가 적용 대상이 아니다.
    적용 대상으로 잘못 잡으면 "도구를 호출하라"고 유도해 접수 확인 없이
    바로 Step2로 넘어가게 만드는 실패가 실측으로 확인됐다(스모크 테스트,
    세션 f4081a6a)."""
    state = _base_state()
    state["disclosed_stage"] = {_STAGE_DOMAIN: 4}
    state["messages"] = [
        HumanMessage(content="공사는 어떻게 진행돼?"),
        AIMessage(content="다음은 Step 2(공사) 단계입니다. 착공신고를 먼저 진행하셔야 해요."),
    ]
    update = guard_node(state)
    assert not any(
        isinstance(m, ToolMessage) and m.name == "_answer_guard" for m in update.get("messages", [])
    )


def test_construction_guide_called_suppresses_gap_violation():
    """get_construction_guide를 실제로 호출한 뒤라면(같은 턴이든 이전 턴이든)
    Step 2 내용을 언급해도 더 이상 걸리지 않아야 한다 - 정상 경로까지
    막으면 안 됨."""
    state = _base_state()
    state["disclosed_stage"] = {_STAGE_DOMAIN: 4}
    state["task_progress"] = {"application_submitted": True}
    call_id = "c1"
    state["messages"] = [
        HumanMessage(content="공사는 어떻게 진행돼?"),
        AIMessage(content="", tool_calls=[{"name": "get_construction_guide", "args": {}, "id": call_id}]),
        ToolMessage(content="[착공ᆞ공사 단계 안내]...", tool_call_id=call_id, name="get_construction_guide"),
        AIMessage(content="다음은 Step 2(공사) 단계입니다. 착공신고를 먼저 진행하셔야 해요."),
    ]
    update = guard_node(state)
    assert not any(
        isinstance(m, ToolMessage) and m.name == "_answer_guard" for m in update.get("messages", [])
    )


def test_construction_guide_gap_ignores_unrelated_topics():
    """Step2 관련 키워드가 아예 없는(다른 도메인) 답변은 이 체크가 무시해야
    한다 - Step2를 아직 안 물었는데 강제로 끼워넣으면 과잉 개입이 된다."""
    state = _base_state()
    state["disclosed_stage"] = {_STAGE_DOMAIN: 4}
    state["task_progress"] = {"application_submitted": True}
    state["messages"] = [
        HumanMessage(content="간판은 신고 대상인가요?"),
        AIMessage(content="네, 벽면이용간판은 신고 대상입니다."),
    ]
    update = guard_node(state)
    assert not any(
        isinstance(m, ToolMessage) and m.name == "_answer_guard" for m in update.get("messages", [])
    )


def test_force_construction_guide_when_ai_just_proposed_step2():
    """AI가 방금 "Step 2(공사 단계)" 안내를 제안했고 사용자가 그에 응답하는
    턴이면 tool_choice 강제 대상이다 - 세션 5db53ccb의 실패 패턴 그대로."""
    state = _base_state()
    state["disclosed_stage"] = {_STAGE_DOMAIN: 4}
    state["task_progress"] = {"application_submitted": True}
    state["messages"] = [
        HumanMessage(content="서류ᆞ사전진단 다 끝냈어."),
        AIMessage(content="완료하셨군요! 다음은 Step 2(공사 단계)입니다. 안내해 드릴까요?"),
        HumanMessage(content="응 알려줘"),
    ]
    assert _should_force_construction_guide(state) is True


def test_no_force_before_application_submitted_even_if_ai_proposed_step2():
    """2026-08-03 회귀 - Step1-4까지 disclosed=4여도 신청ᆞ접수 확인이 아직
    안 됐으면, AI가 스스로 "Step 2(공사 단계)"를 언급했더라도 강제 호출
    대상이 아니다 - 강제해버리면 접수 확인 없이 곧장 Step2로 넘어가는
    실패가 실측으로 확인됐다(스모크 테스트, 세션 f4081a6a)."""
    state = _base_state()
    state["disclosed_stage"] = {_STAGE_DOMAIN: 4}
    state["messages"] = [
        HumanMessage(content="서류ᆞ사전진단 다 끝냈어."),
        AIMessage(content="완료하셨군요! 다음은 Step 2(공사 단계)입니다. 안내해 드릴까요?"),
        HumanMessage(content="응 알려줘"),
    ]
    assert _should_force_construction_guide(state) is False


def test_no_force_when_construction_guide_already_shown():
    """이미 한 번 호출됐으면(플래그 세팅됨) 더 강제할 필요 없다."""
    state = _base_state()
    state["disclosed_stage"] = {_STAGE_DOMAIN: 4, "construction_guide": 1}
    state["task_progress"] = {"application_submitted": True}
    state["messages"] = [
        HumanMessage(content="공사는 어떻게 진행돼?"),
        AIMessage(content="다음은 Step 2(공사 단계)입니다. 안내해 드릴까요?"),
        HumanMessage(content="응"),
    ]
    assert _should_force_construction_guide(state) is False


def test_no_force_when_step1_not_finished():
    """Step 1이 아직 안 끝났으면(disclosed<4) 강제 대상이 아니다."""
    state = _base_state()
    state["disclosed_stage"] = {_STAGE_DOMAIN: 1}
    state["messages"] = [
        HumanMessage(content="공사는 어떻게 돼?"),
        AIMessage(content="Step 2(공사 단계)는 나중에 안내해 드릴게요."),
        HumanMessage(content="응"),
    ]
    assert _should_force_construction_guide(state) is False


def test_no_force_on_unrelated_topic_even_if_step1_done():
    """직전 AI 메시지가 Step2를 제안한 게 아니라 완전히 다른 주제(예: 간판)
    였다면, 사용자가 뭘 답하든 강제하면 안 된다 - 무관한 질문까지 공사
    안내로 강제 전환되는 부작용을 막는 게 이 함수의 핵심 존재 이유."""
    state = _base_state()
    state["disclosed_stage"] = {_STAGE_DOMAIN: 4}
    state["task_progress"] = {"application_submitted": True}
    state["messages"] = [
        HumanMessage(content="간판은 신고 대상인가요?"),
        AIMessage(content="네, 벽면이용간판은 신고 대상입니다."),
        HumanMessage(content="응 알겠어"),
    ]
    assert _should_force_construction_guide(state) is False


def test_no_force_when_last_message_is_not_human():
    """방금 도구가 호출된 직후(마지막 메시지가 HumanMessage가 아님)라면
    이번 판단 대상이 아니다 - agent_node가 다음 LLM 호출 전에 판단하는
    시점 자체가 아직 아님."""
    state = _base_state()
    state["disclosed_stage"] = {_STAGE_DOMAIN: 4}
    state["task_progress"] = {"application_submitted": True}
    state["messages"] = [
        HumanMessage(content="응 알려줘"),
        AIMessage(content="다음은 Step 2(공사 단계)입니다. 안내해 드릴까요?"),
    ]
    assert _should_force_construction_guide(state) is False


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


def test_correction_omission_triggers_retry():
    """idea_notes.md 스펙 그대로 재현: 유저가 이미 기록된 수치를 정정하는
    발화를 했는데(기존 size_sqm=85, "아까 85㎡라고 했는데 사실 100㎡야")
    AI가 record_case_facts를 다시 호출하지 않고 답만 하면 재시도를
    유도해야 한다 - 옛 값이 남으면 잘못된 판정으로 이어짐."""
    state = _base_state()
    state["case_facts"] = {"_classified": True, "size_sqm": 85, "floors": 2}
    state["messages"] = [
        HumanMessage(content="아까 85㎡라고 했는데 사실 100㎡야"),
        AIMessage(content="네, 확인했습니다."),
    ]
    update = guard_node(state)
    guard_msgs = [
        m for m in update.get("messages", []) if isinstance(m, ToolMessage) and m.name == "_answer_guard"
    ]
    assert guard_msgs, "정정 발화인데도 record_case_facts 미호출이 재시도로 안 잡힘"
    assert "record_case_facts" in guard_msgs[0].content


def test_correction_omission_suppressed_when_tool_called():
    """같은 정정 발화라도, AI가 실제로 record_case_facts를 다시 호출했다면
    (merge_facts가 알아서 필드를 덮어씀) 위반이 아니다 - 정상 경로를 막으면
    안 된다."""
    state = _base_state()
    state["case_facts"] = {"_classified": True, "size_sqm": 85, "floors": 2}
    call_id = "c1"
    state["messages"] = [
        HumanMessage(content="아까 85㎡라고 했는데 사실 100㎡야"),
        AIMessage(
            content="정정 감사합니다.",
            tool_calls=[{"name": "record_case_facts", "args": {"size_sqm": 100}, "id": call_id}],
        ),
        ToolMessage(content="기록됨.", tool_call_id=call_id, name="record_case_facts"),
    ]
    update = guard_node(state)
    assert not any(
        isinstance(m, ToolMessage) and m.name == "_answer_guard" for m in update.get("messages", [])
    )


def test_correction_omission_ignores_reconfirmation():
    """언급된 숫자가 이미 기록된 값과 같으면(재확인일 뿐 정정이 아님) 위반이
    아니다 - "아까 85㎡라고 했잖아" 같은 재확인까지 오탐하면 안 된다."""
    state = _base_state()
    state["case_facts"] = {"_classified": True, "size_sqm": 85}
    state["messages"] = [
        HumanMessage(content="아까 85㎡라고 했잖아"),
        AIMessage(content="네, 맞습니다."),
    ]
    update = guard_node(state)
    assert not any(
        isinstance(m, ToolMessage) and m.name == "_answer_guard" for m in update.get("messages", [])
    )


def test_correction_omission_ignores_without_prior_numeric_facts():
    """정정할 기존 수치 자체가 없으면(case_facts에 숫자 필드가 하나도 없음)
    키워드ᆞ숫자가 있어도 위반이 아니다."""
    state = _base_state()
    state["case_facts"] = {"_classified": False}
    state["messages"] = [
        HumanMessage(content="아니 정정할게, 85㎡야"),
        AIMessage(content="네, 알겠습니다."),
    ]
    update = guard_node(state)
    assert not any(
        isinstance(m, ToolMessage) and m.name == "_answer_guard" for m in update.get("messages", [])
    )
