"""classify_fire_safety()(src/agents/fire_safety.py) 연면적 경계값 회귀 테스트.

임계값은 하드코딩하지 않고 get_thresholds()에서 그대로 읽어와 사용한다 -
permit_thresholds.yaml 값이 나중에 법령 개정으로 바뀌어도 이 테스트가 옛날
숫자와 어긋나 실패하는 게 아니라, 코드가 실제로 사용하는 최신 값 기준으로
경계 동작(><=)이 맞는지를 계속 검증하게 하기 위함.
"""
from __future__ import annotations

from src.agents.fire_safety import classify_fire_safety
from src.agents.legal_data import get_thresholds

THRESHOLDS = get_thresholds()["소방시설_연면적기준_sqm"]


def test_missing_facts_is_unclassified():
    assert classify_fire_safety({}) is None
    assert classify_fire_safety({"size_sqm": 100}) is None  # is_large_store_tenant 없음
    assert classify_fire_safety({"is_large_store_tenant": False}) is None  # size_sqm 없음


def test_below_all_thresholds_returns_empty_list_not_none():
    """빈 리스트는 '해당 없음'이라는 유효한 판정이지 미분류가 아니다."""
    result = classify_fire_safety({"size_sqm": THRESHOLDS["소화기구"] - 1, "is_large_store_tenant": False})
    assert result == []


def test_fire_extinguisher_threshold_boundary():
    boundary = THRESHOLDS["소화기구"]
    assert "소화기구" in classify_fire_safety({"size_sqm": boundary, "is_large_store_tenant": False})
    assert "소화기구" not in classify_fire_safety({"size_sqm": boundary - 1, "is_large_store_tenant": False})


def test_alarm_threshold_boundary():
    boundary = THRESHOLDS["비상경보설비"]
    result = classify_fire_safety({"size_sqm": boundary, "is_large_store_tenant": False})
    assert "비상경보설비" in result
    assert "소화기구" in result  # 더 낮은 임계값도 당연히 함께 충족


def test_detector_threshold_boundary():
    boundary = THRESHOLDS["자동화재탐지설비"]
    result = classify_fire_safety({"size_sqm": boundary, "is_large_store_tenant": False})
    assert "자동화재탐지설비" in result


def test_sprinkler_threshold_all_four_required_items_present():
    boundary = THRESHOLDS["간이스프링클러설비"]
    result = classify_fire_safety({"size_sqm": boundary, "is_large_store_tenant": False})
    assert set(result) == {"소화기구", "비상경보설비", "자동화재탐지설비", "간이스프링클러설비"}


def test_large_store_tenant_adds_kitchen_extinguisher_regardless_of_size():
    """대규모점포 입점 여부는 연면적과 독립적인 별도 조건 - 아주 작은
    면적이어도 이 항목만 추가된다."""
    result = classify_fire_safety({"size_sqm": 1, "is_large_store_tenant": True})
    assert result == ["상업용 주방자동소화장치"]


def test_large_store_tenant_combined_with_full_size_thresholds():
    boundary = THRESHOLDS["간이스프링클러설비"]
    result = classify_fire_safety({"size_sqm": boundary, "is_large_store_tenant": True})
    assert set(result) == {
        "소화기구", "비상경보설비", "자동화재탐지설비", "간이스프링클러설비", "상업용 주방자동소화장치",
    }
