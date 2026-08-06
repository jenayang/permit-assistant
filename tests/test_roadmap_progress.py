"""cascade_construction_progress 단위 테스트.

사용승인은 법적으로 착공신고ᆞ시공이 끝나야만 가능한 절차라, use_approval=True를
기록할 때 앞단계(construction_notice/construction)도 자동으로 True 채워야 한다
(2026-08-04 - "사용승인 완료했는데 착공신고ᆞ시공 체크박스가 그대로"라는 신고).
"""
from __future__ import annotations

from src.agents.roadmap_progress import TASK_PROGRESS_FIELDS, cascade_construction_progress


def test_new_role_fields_registered():
    assert "uses_agency" in TASK_PROGRESS_FIELDS
    assert "architect_selected" in TASK_PROGRESS_FIELDS
    assert "interior_only" in TASK_PROGRESS_FIELDS


def test_use_approval_cascades_to_earlier_fields():
    result = cascade_construction_progress({"use_approval": True})
    assert result == {"use_approval": True, "construction": True, "construction_notice": True}


def test_construction_cascades_to_construction_notice_only():
    result = cascade_construction_progress({"construction": True})
    assert result == {"construction": True, "construction_notice": True}
    assert "use_approval" not in result


def test_explicit_false_is_not_overwritten_by_cascade():
    result = cascade_construction_progress({"construction_notice": False, "use_approval": True})
    assert result["construction_notice"] is False
    assert result["construction"] is True
    assert result["use_approval"] is True


def test_unrelated_fields_pass_through_untouched():
    result = cascade_construction_progress({"hygiene_education": True})
    assert result == {"hygiene_education": True}


def test_false_value_does_not_trigger_cascade():
    result = cascade_construction_progress({"use_approval": False})
    assert result == {"use_approval": False}
