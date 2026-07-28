"""classify_food_business()(src/agents/food_safety.py) 단락평가(short-circuit)
회귀 테스트.

이 함수는 2026-07-24에 이미 한 번 "네 필드가 전부 모여야만 판정"하던 걸
"확정되는 순서대로 단락 평가"하도록 고친 이력이 있다(docstring 참고) -
그 수정이 여전히 유효한지 회귀로 고정한다. primarily_bakery가
sells_ready_made_only보다 먼저 검사되므로, sells_ready_made_only=True가
primarily_bakery를 모르는 상태에서도 즉시 확정되는 것까지 포함해서 검증한다
(코드가 실제로 그렇게 동작함 - 의도된 순서 의존성).
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


def test_ready_made_only_short_circuits_even_if_bakery_unknown():
    """primarily_bakery가 sells_ready_made_only보다 먼저 검사되지만, 아직
    모르는(None) 상태에서 sells_ready_made_only=True를 알면 그 자리에서
    바로 확정된다(코드의 실제 분기 순서 - 의도된 동작을 회귀로 고정)."""
    assert classify_food_business({
        "serves_food": True, "sells_ready_made_only": True,
    }) == "즉석판매제조가공업"


def test_alcohol_determines_general_vs_rest_restaurant():
    base = {"serves_food": True, "primarily_bakery": False, "sells_ready_made_only": False}
    assert classify_food_business({**base, "serves_alcohol": True}) == "일반음식점영업"
    assert classify_food_business({**base, "serves_alcohol": False}) == "휴게음식점영업"


def test_missing_alcohol_field_is_unclassified():
    assert classify_food_business({
        "serves_food": True, "primarily_bakery": False, "sells_ready_made_only": False,
    }) is None


def test_missing_bakery_and_ready_made_fields_is_unclassified():
    assert classify_food_business({"serves_food": True}) is None
