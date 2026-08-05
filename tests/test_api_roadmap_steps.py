"""ProjectStatusResponse.steps(src/api.py, Phase 4) 스모크 테스트.

compute_project_status()가 이제 roadmap_model.compute_roadmap_steps()를
camelCase dict로 실어 "steps" 키에 내려보낸다(src/project_status.py 참고) -
그 dict가 StepOut/SubstepOut 스키마로 실제로 검증되고, FastAPI 응답
직렬화(response_model_by_alias=True 기본값)에서 camelCase 키로 나가는지
확인한다. LLM 호출 없는 순수 함수 조합만 테스트한다.
"""
from __future__ import annotations

from src.api import ProjectStatusResponse
from src.project_status import compute_project_status


def test_steps_field_has_nine_entries_in_order():
    data = compute_project_status({})
    resp = ProjectStatusResponse(**data)
    assert [s.index for s in resp.steps] == list(range(1, 10))
    assert resp.steps[0].title == "주소 및 건축물 현황 확인"
    assert resp.steps[8].title == "간판신고 및 오픈 준비"


def test_steps_serialize_with_camel_case_keys():
    data = compute_project_status({
        "case_facts": {"act_type": "신축"},
        "permit_result": {"permit_type": "건축신고"},
        "task_progress": {},
    })
    resp = ProjectStatusResponse(**data)
    dumped = resp.model_dump(by_alias=True)
    pre_review_step = dumped["steps"][3]  # "인허가 사전 검토"
    last_substep = pre_review_step["substeps"][-1]
    assert "checkKey" not in last_substep or last_substep["checkKey"] is None
    assert "lockReason" in last_substep
    assert "notApplicable" in last_substep
    assert "naReason" in last_substep
    # snake_case 키는 응답에 남아있지 않아야 한다(카멜케이스로만 나감).
    assert "check_key" not in last_substep
    assert "not_applicable" not in last_substep


def test_steps_field_does_not_break_existing_fields():
    # 기존 5개 필드(progress/completed/construction_track/startup_track/
    # summary/requires_architect)는 steps 추가와 무관하게 그대로여야 한다
    # (breaking change 없음, Phase 4 성공 기준).
    data = compute_project_status({})
    resp = ProjectStatusResponse(**data)
    assert resp.progress == 0
    assert resp.completed == []
    assert resp.construction_track.current_step == "건축 유형 확인"
