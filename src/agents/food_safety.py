"""Food Safety Agent - 식품위생법상 영업 종류 판정(규칙 기반).

permit.py의 classify_case와 동일한 원칙: 휴게음식점ᆞ일반음식점 등 식품접객업의
세부 종류는 식품위생법 시행령 제21조에 명시된 객관적 기준(주로 조리ᆞ판매하는
품목, 음주행위 허용 여부)으로 정해지므로 LLM 판단에 맡기지 않고
classify_food_business()가 파이썬 규칙으로 결정론적으로 계산한다(원문 대조:
data/laws/식품위생법 시행령(대통령령)(제35811호)(20251001).pdf, 제21조).

이 챗봇의 타깃(카페ᆞ음식점ᆞ베이커리 등 소규모 창업)과 거리가 먼
단란주점ᆞ유흥주점ᆞ위탁급식영업(제21조8호다ᆞ라ᆞ마)은 범위 밖으로 둔다 -
필요해지면 classify_food_business에 분기만 추가하면 된다(REQUIRED_FIELDS_FOOD도
같이 확장 필요).

건축(permit.py)과 식품위생(이 파일)은 서로 다른 법령 도메인이라 판정에 쓰는
사실(facts)도 독립적이다 - 카페 창업처럼 두 판정이 동시에 필요한 경우
agent.py의 상태에 case_facts/food_facts가 각각 별도 채널로 누적된다(둘 다
None 아닌 값만 덮어쓰는 merge_facts 리듀서 재사용 예정).
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# classify_food_business가 판정에 참조하는 사실 목록. 하나라도 비어있으면(None)
# 아직 분류 안 함 - agent.py 그래프가 이걸로 "정보 충분?" 판단(permit.py의
# REQUIRED_FIELDS와 동일한 역할).
REQUIRED_FIELDS_FOOD: list[str] = [
    "serves_food", "sells_ready_made_only", "primarily_bakery", "serves_alcohol",
]


def _missing_fields_food(facts: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS_FOOD if facts.get(f) is None]


def classify_food_business(facts: dict) -> str | None:
    """food_facts로 식품위생법 시행령 제21조상 영업 종류를 결정. 정보 부족하면 None.

    facts에서 쓰는 키(전부 bool):
      serves_food: 음식류를 조리ᆞ판매하는 업종인지. False면 식품위생법상
        영업신고 대상 자체가 아님(예: 사무실ᆞ일반 소매업ᆞ미용실).
      sells_ready_made_only: 조리 없이 이미 완성된 식품(포장 완제품)만
        최종소비자에게 판매하는지 - 제21조2호 즉석판매제조ᆞ가공업 기준.
      primarily_bakery: 주로 빵ᆞ떡ᆞ과자를 제조ᆞ판매하는지 - 제21조8호바
        제과점영업 기준(직접 조리ᆞ제조해 판매하는 베이커리는 여기 해당,
        sells_ready_made_only보다 우선 판단).
      serves_alcohol: 식사와 함께 음주행위(주류 판매)가 허용되는지 - 이게
        휴게음식점(불허, 제21조8호가)과 일반음식점(허용, 제21조8호나)을
        가르는 핵심 기준.

    네 값이 전부 모여야 판정한다(미확정 상태로 추측 판정하지 않음 -
    _missing_fields_food가 먼저 걸러줌).
    """
    if _missing_fields_food(facts):
        return None

    if not facts["serves_food"]:
        return "해당없음"  # 식품위생법 영업신고 대상 아님

    if facts["sells_ready_made_only"]:
        return "즉석판매제조가공업"  # 시행령 제21조2호

    if facts["primarily_bakery"]:
        return "제과점영업"  # 시행령 제21조8호바

    return "일반음식점영업" if facts["serves_alcohol"] else "휴게음식점영업"  # 시행령 제21조8호나ᆞ가


def food_business_message(business_type: str) -> str:
    """classify_food_business()의 판정 결과를 LLM이 답변에 반영할 안내 문구로
    변환. procedure_stage_message(permit.py)와 동일한 역할 - agent.py의
    classify 노드가 분류 결과를 답변에 반영할 때 참고용으로 쓴다.
    """
    if business_type == "해당없음":
        return "식품위생법상 별도 영업신고 대상이 아닙니다(음식류를 조리ᆞ판매하지 않는 업종)."
    return f"식품위생법상 영업 종류 판정: {business_type}. 관할 보건소에 영업신고가 필요합니다."


@tool
def record_food_facts(
    serves_food: bool | None = None,
    sells_ready_made_only: bool | None = None,
    primarily_bakery: bool | None = None,
    serves_alcohol: bool | None = None,
) -> str:
    """식품위생법상 영업 종류 판정에 필요한 사용자 업종 정보를 알게 되는 대로 기록하세요.

    - 이번 턴에 새로 알게 된 필드만 채우고 나머지는 생략(None)하세요. 여러 턴에
      걸쳐 나눠서 호출해도 이전에 기록한 값 위에 누적됩니다.
    - 영업 종류(휴게음식점/일반음식점 등) 판정은 여기서 계산 안 합니다 - 정보가
      충분히 모이면 시스템이 자동으로(규칙 기반) 판정하니, 사용자에게 "휴게음식점
      대상인가요 일반음식점 대상인가요?" 같은 질문은 하지 마세요. 사용자가 실제로
      답할 수 있는 사실(음주 판매 여부ᆞ조리 여부 등)만 물어보고 기록하세요.
    - 카페ᆞ식당처럼 음식류를 조리ᆞ판매하는 업종이 명백하면 serves_food=True를
      먼저 기록하고, 사무실ᆞ미용실처럼 식품위생법과 무관한 업종이면
      serves_food=False만 기록해도 곧바로 "해당없음"으로 판정됩니다.

    Args:
        serves_food: 음식류를 조리ᆞ판매하는 업종인지 여부
        sells_ready_made_only: 조리 없이 완제품만 최종소비자에게 판매하는지
        primarily_bakery: 주로 빵ᆞ떡ᆞ과자를 제조ᆞ판매하는지
        serves_alcohol: 식사와 함께 음주행위(주류 판매)가 허용되는지
    """
    logger.info("[도구] record_food_facts(%r)", {
        k: v for k, v in locals().items() if v is not None
    })
    return "기록됨."
