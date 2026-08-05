"""guard_node/current_step_directive/domain_timing_directive 시나리오 테스트.

LLM을 호출하지 않는다 - 전부 상태(AgentState)만 보고 결정론적으로 동작하는
함수라, 실제 대화 턴을 messages 시퀀스로 흉내내는 것만으로 정확히 재현
가능하다.

2026-08-05: [Step 1-N]/[Step 2-N] 텍스트 마커 + disclosed_stage 페이싱을
current_step_directive(roadmap_model.compute_roadmap_steps() 기반)로
교체하면서, 그 마커 체계만 검증하던 테스트들(_max_allowed_stage/
_mentioned_stages/permit_phase_directive/_should_force_construction_guide 등)을
전부 제거하고 새 함수 테스트로 바꿨다. 마커와 무관하게 유효했던 테스트
(건축물대장 시설군 자동 매핑, 수치 정정 누락 감지, 도메인 실행 시점 안내)는
그대로 유지한다.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.classify import _derive_ledger_facility_group
from src.guard import guard_node
from src.roadmap import current_step_directive, domain_timing_directive


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


def _base_state() -> dict:
    return {
        "messages": [],
        "case_facts": {"_classified": True},
        "permit_synthesis": {},
        "task_progress": {},
    }


# --- current_step_directive ---------------------------------------------

def test_current_step_directive_none_before_classification():
    state = {"case_facts": {}}
    assert current_step_directive(state) is None


def test_current_step_directive_lists_construction_step_checklist():
    """신축ᆞ건축사 의무 대상 케이스는 Step1~2가 이미 채워져 있으면 Step3
    (건축사사무소 선정 및 도면 작성)이 현재 단계로 잡히고, 그 substep들이
    field 참조와 함께 나열된다."""
    state = _base_state()
    state["case_facts"] = {
        "_classified": True, "act_type": "신축", "address": "서울시 종로구",
        "size_sqm": 100, "floors": 2, "land_zone": "기타", "ownership": "소유자",
    }
    state["permit_result"] = {"permit_type": "신고"}
    directive = current_step_directive(state)
    assert directive is not None
    assert "Step 3: 건축사사무소 선정 및 도면 작성" in directive
    assert "현장 방문" in directive
    assert "record_task_progress(site_visit_done=True)" in directive


def test_current_step_directive_skips_done_and_locked_substeps():
    """건축사사무소를 이미 선정했으면 그 항목은 다시 안내 목록에 안 나오고,
    아직 공사 전이라 소방ᆞ간판처럼 잠긴 항목은 나오지 않는다."""
    state = _base_state()
    state["case_facts"] = {
        "_classified": True, "act_type": "신축", "size_sqm": 100, "floors": 2,
    }
    state["permit_result"] = {"permit_type": "신고"}
    state["task_progress"] = {"architect_selected": True}
    directive = current_step_directive(state)
    assert "건축사사무소 의뢰 및 계약 체결" not in directive


def test_current_step_directive_includes_startup_track_once_unlocked():
    """공사(시공)가 끝나면 병행 가능 트랙(Step8~9)의 잠금이 풀린 항목이
    같이 나온다."""
    state = _base_state()
    state["case_facts"] = {"_classified": True, "act_type": "신축"}
    state["permit_result"] = {"permit_type": "신고"}
    state["food_result"] = "일반음식점"
    state["signage_result"] = "신고"
    state["task_progress"] = {"construction": True, "use_approval": True}
    directive = current_step_directive(state)
    assert directive is not None
    assert "[병행 가능 단계] Step 8" in directive or "[병행 가능 단계] Step 9" in directive


def test_current_step_directive_attaches_step_guidance():
    """Step3~7 범위에서는 STEP_GUIDANCE의 배경 설명도 함께 붙는다."""
    state = _base_state()
    state["case_facts"] = {
        "_classified": True, "act_type": "신축", "address": "서울시 종로구",
        "size_sqm": 100, "floors": 2, "land_zone": "기타", "ownership": "소유자",
    }
    state["permit_result"] = {"permit_type": "신고"}
    directive = current_step_directive(state)
    assert "Step 3: 건축사사무소 선정 및 도면 작성" in directive
    assert "건축사사무소 설계 의무 여부는" in directive


# --- domain_timing_directive ----------------------------------------------
# 2026-08-05: "간판(Step9)을 먼저 물어봐도 판정은 즉답하되, 실행 시점
# (병렬 가능ᆞ선행 단계)도 같이 안내하라"는 사용자 피드백에서 나온 함수 -
# 판정 여부와 무관하게 아무것도 없으면 None, 판정된 도메인이 있으면
# 시공ᆞ사용승인 완료 여부에 따라 문구가 갈린다.
def test_domain_timing_directive_none_when_no_domain_results():
    state = _base_state()
    assert domain_timing_directive(state) is None


def test_domain_timing_directive_fire_signage_locked_before_construction():
    state = _base_state()
    state["fire_result"] = ["소화기구"]
    state["signage_result"] = "신고"
    directive = domain_timing_directive(state)
    assert directive is not None
    assert "시공" in directive and "끝난 뒤" in directive


def test_domain_timing_directive_fire_signage_unlocked_after_construction():
    state = _base_state()
    state["fire_result"] = ["소화기구"]
    state["signage_result"] = "신고"
    state["task_progress"] = {"construction": True}
    directive = domain_timing_directive(state)
    assert "지금 바로 설치를 진행" in directive


def test_domain_timing_directive_food_report_locked_before_use_approval():
    state = _base_state()
    state["food_result"] = "일반음식점"
    directive = domain_timing_directive(state)
    assert "사용승인이 끝난 뒤" in directive
    # 위생교육은 언제든 가능하다는 안내가 항상 같이 붙는다(병렬 진행 가능).
    assert "위생교육" in directive and "병렬 진행 가능" in directive


def test_domain_timing_directive_food_report_unlocked_after_use_approval():
    state = _base_state()
    state["food_result"] = "일반음식점"
    state["task_progress"] = {"use_approval": True}
    directive = domain_timing_directive(state)
    assert "지금 바로" in directive and "신고 접수를 진행" in directive


# --- guard_node: 수치 정정 누락 감지 ---------------------------------------

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
