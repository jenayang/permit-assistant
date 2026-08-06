"""src/classify.py의 _classify_permit(classify_case를 ClassificationOutput으로
감싸는 배관 레이어) 회귀 테스트.

_walk_procedures(permit.py)는 ownership이 이미 알려져 있으면 owner_check
분기를 자동으로 건너뛰는데, _classify_permit이 set_procedure_stage에 넘기는
node_id는 그 로직을 안 타서 항상 트리 root("owner_check")로 고정돼 있었다 -
그 결과 LLM에게 "다음을 사용자에게 물어보세요: 소유자이신가요, 임차인이신가요?"를
매번 지시해서, 사용자가 폼ᆞ채팅에서 이미 답한 소유/임차 여부를 판정 직후
다시 묻는 버그로 이어졌다(2026-08-06 재신고, 세션 75280711)."""
from __future__ import annotations

from src.classify import _classify_permit


def test_ownership_known_skips_owner_check_question():
    facts = {
        "land_zone": "기타", "current_facility_group": 8, "act_type": "용도변경",
        "desired_facility_group": 7, "ownership": "임차인",
    }
    out = _classify_permit(facts)
    assert out.tool_args["node_id"] != "owner_check"
    assert "물어보세요" not in out.tool_message
    assert "임대차계약서" in out.tool_message


def test_ownership_unknown_still_asks_owner_check_question():
    facts = {
        "land_zone": "기타", "current_facility_group": 8, "act_type": "용도변경",
        "desired_facility_group": 7,
    }
    out = _classify_permit(facts)
    assert out.tool_args["node_id"] == "owner_check"
    assert "소유자이신가요, 임차인이신가요" in out.tool_message
