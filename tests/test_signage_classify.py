"""classify_signage()(src/agents/signage.py) 경계값ᆞ단락평가(short-circuit)
회귀 테스트.

2026-07-28: permit.py의 classify_case에서 "필요한 값 일부만으로 이미 결론이
나는데도 무관한 필드까지 요구하며 계속 되묻는" 버그를 찾아 고친 뒤, 같은
패턴이 이 파일(벽면이용간판ᆞ돌출간판)에도 있는 걸 감사로 발견해 같이
고쳤다. 이 파일은 그 수정이 두 방향 모두 안전한지 검증한다:
- "허가"(더 엄격한 결과) 쪽으로의 조기 확정은 항상 허용 - 다른 필드가
  뭐든 결과를 더 엄격하게 뒤집을 수 없으므로 안전하다.
- "신고"/"불필요"(더 완화된 결과) 쪽은 부분 정보만으로 성급히 확정하면
  안 된다 - 나중에 실제로는 허가 대상이었던 게 드러날 위험이 있어서다.
  단, 돌출간판의 완화 조건 세 개는 서로 독립적인 OR 조건이라 그중 하나만
  확인돼도 안전하게 조기 확정할 수 있다(벽면이용간판과 다른 점).
"""
from __future__ import annotations

from src.agents.signage import classify_signage


# --- 벽면이용간판 ---------------------------------------------------------

def test_wall_sign_length_alone_short_circuits_to_permit():
    """length_m>=10이면 다른 필드를 몰라도 즉시 허가(2026-07-28 수정)."""
    assert classify_signage({"sign_type": "벽면이용간판", "length_m": 15}) == "허가"
    assert classify_signage({"sign_type": "벽면이용간판", "length_m": 10}) == "허가"  # 경계 포함


def test_wall_sign_length_below_boundary_alone_is_insufficient():
    assert classify_signage({"sign_type": "벽면이용간판", "length_m": 9.99}) is None


def test_wall_sign_floor_and_third_party_short_circuits_to_permit():
    """floor>=4 + is_third_party_ad=True면 length_m을 몰라도 즉시 허가."""
    assert classify_signage({
        "sign_type": "벽면이용간판", "floor": 4, "is_third_party_ad": True,
    }) == "허가"


def test_wall_sign_area_alone_does_not_short_circuit_to_notice():
    """area_sqm>=5만 알고 length_m/floor/is_third_party_ad를 모르면, 그
    floor+타사광고 조합으로 실제로는 허가일 수도 있으니 신고로 성급히
    단정하면 안 된다 - 여전히 None(정보 부족)이어야 한다."""
    assert classify_signage({"sign_type": "벽면이용간판", "area_sqm": 10}) is None


def test_wall_sign_notice_via_area_needs_full_info():
    facts = {"sign_type": "벽면이용간판", "length_m": 5, "floor": 1, "is_third_party_ad": False, "area_sqm": 5}
    assert classify_signage(facts) == "신고"


def test_wall_sign_notice_via_floor_without_third_party_ad():
    facts = {"sign_type": "벽면이용간판", "length_m": 5, "floor": 4, "is_third_party_ad": False, "area_sqm": 1}
    assert classify_signage(facts) == "신고"


def test_wall_sign_exempt_when_all_thresholds_unmet():
    facts = {"sign_type": "벽면이용간판", "length_m": 5, "floor": 1, "is_third_party_ad": False, "area_sqm": 1}
    assert classify_signage(facts) == "허가ᆞ신고 불필요"


# --- 돌출간판 --------------------------------------------------------------

def test_protruding_sign_medical_alone_short_circuits_to_notice():
    assert classify_signage({"sign_type": "돌출간판", "is_medical_or_salon_sign": True}) == "신고"


def test_protruding_sign_low_height_alone_short_circuits_to_notice():
    assert classify_signage({"sign_type": "돌출간판", "height_m": 3}) == "신고"


def test_protruding_sign_small_area_alone_short_circuits_to_notice():
    assert classify_signage({"sign_type": "돌출간판", "area_sqm": 0.5}) == "신고"


def test_protruding_sign_non_exempt_partial_info_is_insufficient():
    assert classify_signage({"sign_type": "돌출간판", "height_m": 10}) is None


def test_protruding_sign_permit_requires_all_three_confirmed():
    facts = {"sign_type": "돌출간판", "height_m": 10, "area_sqm": 5, "is_medical_or_salon_sign": False}
    assert classify_signage(facts) == "허가"


# --- 지주이용간판ᆞ입간판ᆞ현수막(단순 분기, 회귀 확인용) -------------------

def test_pole_sign_boundary():
    assert classify_signage({"sign_type": "지주이용간판", "height_m": 4}) == "허가"
    assert classify_signage({"sign_type": "지주이용간판", "height_m": 3.99}) == "신고"
    assert classify_signage({"sign_type": "지주이용간판"}) is None


def test_standing_sign_always_notice():
    assert classify_signage({"sign_type": "입간판"}) == "신고"


def test_banner_boundary():
    assert classify_signage({"sign_type": "현수막", "display_facility_area_sqm": 30}) == "신고"  # 경계 포함(초과만 허가)
    assert classify_signage({"sign_type": "현수막", "display_facility_area_sqm": 30.01}) == "허가"
    assert classify_signage({"sign_type": "현수막"}) is None


def test_unknown_or_missing_sign_type_returns_none():
    assert classify_signage({}) is None
    assert classify_signage({"sign_type": None}) is None
