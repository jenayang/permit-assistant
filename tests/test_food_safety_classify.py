"""classify_food_business()(src/agents/food_safety.py) 단락평가(short-circuit)
회귀 테스트.

이 함수는 확정되는 순서대로 단락 평가한다(모든 필드가 모여야만 판정하지
않음). 2026-08-03에 "완제품만 판매(제조ᆞ가공 없음)" 취급을 법령에 맞게
고쳤다 - 과거엔 이걸 즉석판매제조ᆞ가공업(제조ᆞ가공이 요건)으로 잘못 판정했으나,
지금은 manufactures_or_cooks=False + 영업장 면적으로 기타식품판매업(300㎡ 이상,
시행규칙 제39조) 또는 신고대상제외(300㎡ 미만)로 가른다.
"""
from __future__ import annotations

from src.agents.food_safety import classify_food_business


def test_unknown_serves_food_is_unclassified():
    assert classify_food_business({}) is None


def test_not_food_service_short_circuits_regardless_of_other_fields():
    assert classify_food_business({"serves_food": False}) == "해당없음"
    # 다른 필드가 어떻든 무관 - serves_food=False만으로 확정.
    assert classify_food_business({
        "serves_food": False, "serves_alcohol": True, "primarily_bakery": True,
    }) == "해당없음"


def test_bakery_short_circuits_before_other_fields_known():
    assert classify_food_business({"serves_food": True, "primarily_bakery": True}) == "제과점영업"


def test_finished_products_large_store_is_etc_food_sales():
    """직접 제조ᆞ가공 없이 완제품만 판매 + 영업장 300㎡ 이상 → 기타식품판매업
    (시행령 제21조5호나6ᆞ시행규칙 제39조)."""
    assert classify_food_business({
        "serves_food": True, "manufactures_or_cooks": False, "store_area_sqm": 350,
    }) == "기타식품판매업"
    # 정확히 300㎡도 "이상"이라 신고 대상.
    assert classify_food_business({
        "serves_food": True, "manufactures_or_cooks": False, "store_area_sqm": 300,
    }) == "기타식품판매업"


def test_finished_products_small_store_is_not_reportable():
    """완제품만 판매 + 300㎡ 미만 → 어느 영업 종류에도 미해당(신고대상제외)."""
    assert classify_food_business({
        "serves_food": True, "manufactures_or_cooks": False, "store_area_sqm": 40,
    }) == "신고대상제외"


def test_finished_products_without_area_is_unclassified():
    """완제품만 판매인데 면적을 아직 모르면 판정 불가(300㎡ 경계를 못 가름)."""
    assert classify_food_business({
        "serves_food": True, "manufactures_or_cooks": False,
    }) is None


def test_alcohol_determines_general_vs_rest_restaurant():
    base = {"serves_food": True, "primarily_bakery": False, "manufactures_or_cooks": True}
    assert classify_food_business({**base, "serves_alcohol": True}) == "일반음식점영업"
    assert classify_food_business({**base, "serves_alcohol": False}) == "휴게음식점영업"


def test_cooking_restaurant_not_decided_while_bakery_unknown():
    """조리ᆞ제조하는 음식점 계열이라도 베이커리 여부가 미확정이면 제과점일
    가능성이 남아 음식점으로 단정하지 않는다."""
    assert classify_food_business({
        "serves_food": True, "manufactures_or_cooks": True, "serves_alcohol": False,
    }) is None


def test_missing_alcohol_field_is_unclassified():
    assert classify_food_business({
        "serves_food": True, "primarily_bakery": False, "manufactures_or_cooks": True,
    }) is None


def test_missing_manufacture_and_bakery_fields_is_unclassified():
    assert classify_food_business({"serves_food": True}) is None
