"""permit.py의 절차 트리 탐색(_walk_procedures)ᆞPermitResult 조립
(build_permit_result)ᆞ용도변경 판정 사유 문구(_facility_group_reason) 회귀
테스트.

_facility_group_reason은 특히 과거 실제 버그(2026-07-24)와 직결된다 - LLM이
"번호가 작을수록 상위군"이라는 규칙만 받고 방향(허가/신고 어느 쪽인지)을
스스로 재추론하게 했더니 8→7 같은 이동을 반대로 착각하는 사례가 실측으로
있었고, 그래서 방향 판단ᆞ문구 생성을 전부 코드가 고정 문자열로 만들어
LLM에게 넘기도록 바꿨다. 이 파일은 그 고정 문구의 방향이 실제로 맞는지
검증한다.
"""
from __future__ import annotations

from src.agents.permit import (
    PermitResult,
    _facility_group_reason,
    _walk_procedures,
    build_permit_result,
)


# --- _walk_procedures -------------------------------------------------------

def test_owner_branch_follows_owner_path():
    labels = _walk_procedures("건축허가", "소유자")
    assert labels[0] == "허가 신청 (소유권 증빙)"


def test_tenant_branch_follows_tenant_path():
    labels = _walk_procedures("건축허가", "임차인")
    assert labels[0] == "허가 신청 (토지사용승낙서 등)"


def test_unknown_ownership_defaults_to_first_option():
    """ownership이 None이거나 트리에 없는 값이면 첫 선택지(소유자)로
    기본 진행한다 - 사용자가 아직 소유/임차 여부를 안 밝혀도 절차 안내
    자체는 막히면 안 되기 때문."""
    assert _walk_procedures("건축허가", None)[0] == "허가 신청 (소유권 증빙)"
    assert _walk_procedures("건축허가", "잘못된값")[0] == "허가 신청 (소유권 증빙)"


def test_owner_and_tenant_paths_converge_to_same_remaining_steps():
    owner = _walk_procedures("건축허가", "소유자")
    tenant = _walk_procedures("건축허가", "임차인")
    assert owner[1:] == tenant[1:] == ["착공신고", "착공/시공", "사용승인", "완료"]


def test_non_branch_result_types_ignore_ownership():
    assert _walk_procedures("건축물대장기재변경", "소유자") == _walk_procedures("건축물대장기재변경", None)
    assert _walk_procedures("인허가불필요", "임차인") == ["별도 인허가 없이 진행 가능"]


# --- build_permit_result ----------------------------------------------------

def test_build_permit_result_none_when_unclassified():
    assert build_permit_result({}) is None
    assert build_permit_result({"act_type": "신축"}) is None  # 필드 부족


def test_build_permit_result_matches_classify_case_and_walks_tree():
    facts = {"act_type": "증축", "extension_size_sqm": 85, "ownership": "임차인"}
    result = build_permit_result(facts)
    assert isinstance(result, PermitResult)
    assert result.permit_type == "건축신고"
    assert result.procedures[0] == "신고 (토지사용승낙서 등)"


# --- _facility_group_reason: 용도변경 방향 문구(2026-07-24 회귀 이력) -------

def test_upward_move_smaller_number_produces_permit_direction_text():
    """cur=7→dst=5(번호 감소) = 상위군 이동 = 허가. 문구도 '상위군 이동'
    이라고 정확히 말해야 한다."""
    facts = {"current_facility_group": 7, "desired_facility_group": 5}
    reason = _facility_group_reason("용도변경허가", facts)
    assert "상위군 이동" in reason
    assert "근린생활시설군(7)" in reason and "영업시설군(5)" in reason


def test_downward_move_larger_number_produces_notice_direction_text():
    """cur=5→dst=7(번호 증가) = 하위군 이동 = 신고. 문구도 '하위군 이동'
    이라고 정확히 말해야 한다(반대로 나오면 2026-07-24에 겪은 방향 오판
    버그의 재발)."""
    facts = {"current_facility_group": 5, "desired_facility_group": 7}
    reason = _facility_group_reason("용도변경신고", facts)
    assert "하위군 이동" in reason
    assert "영업시설군(5)" in reason and "근린생활시설군(7)" in reason


def test_same_group_reason_mentions_no_direction():
    facts = {"current_facility_group": 7, "desired_facility_group": 7}
    reason = _facility_group_reason("건축물대장기재변경", facts)
    assert "이동" not in reason
    assert "근린생활시설군(7)" in reason


def test_reason_empty_when_facility_group_missing():
    assert _facility_group_reason("용도변경허가", {}) == ""
    assert _facility_group_reason("용도변경허가", {"current_facility_group": 7}) == ""
