"""Food Safety Agent - 식품위생법상 영업 종류 판정(규칙 기반).

permit.py의 classify_case와 동일한 원칙: 휴게음식점ᆞ일반음식점 등 식품접객업의
세부 종류는 식품위생법 시행령 제21조에 명시된 객관적 기준(주로 조리ᆞ판매하는
품목, 음주행위 허용 여부, 직접 제조ᆞ가공 여부, 영업장 면적)으로 정해지므로
LLM 판단에 맡기지 않고 classify_food_business()가 파이썬 규칙으로 결정론적으로
계산한다(원문 대조: data/laws/식품위생법 시행령(대통령령), 제21조ᆞ제25조,
식품위생법 시행규칙 제39조).

이 챗봇의 타깃(카페ᆞ음식점ᆞ베이커리 등 소규모 창업)과 거리가 먼
단란주점ᆞ유흥주점ᆞ위탁급식영업(제21조8호다ᆞ라ᆞ마)은 범위 밖으로 둔다 -
필요해지면 classify_food_business에 분기만 추가하면 된다.

"완제품만 판매(직접 제조ᆞ가공 없음)"의 취급:
과거엔 이 경우를 즉석판매제조ᆞ가공업(제21조2호)으로 판정했으나, 제21조2호는
"제조ᆞ가공업소에서 직접 최종소비자에게 판매"라 **제조ᆞ가공이 요건**이라
완제품만 되파는 영업과는 맞지 않는다(고시 별표12 예시도 반찬ᆞ죽ᆞ떡가게처럼
직접 만들어 파는 곳이다). 완제품만 판매하는 경우는 시행령 제21조5호나목6)
기타 식품판매업에 해당할 수 있는데, 이는 시행규칙 제39조에 따라 "영업장 면적
300㎡ 이상"인 업소만 신고 대상이다. 따라서 300㎡ 미만에서 완제품만 판매하면
어느 영업 종류에도 해당하지 않아 식품위생법상 별도 영업신고 대상이 아니다.
반면 반찬가게ᆞ떡집처럼 직접 제조ᆞ가공해 포장 판매하는 즉석판매제조ᆞ가공업은
음식점(조리ᆞ제공)과의 법적 구분이 미묘하고 타깃 스코프 밖이라, 이 표만으로는
자동 판정하지 않는다.

건축(permit.py)과 식품위생(이 파일)은 서로 다른 법령 도메인이라 판정에 쓰는
사실(facts)도 독립적이다 - 카페 창업처럼 두 판정이 동시에 필요한 경우
agent.py의 상태에 case_facts/food_facts가 각각 별도 채널로 누적된다.
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 시행규칙 제39조: 기타 식품판매업(영 제21조5호나6)의 "총리령으로 정하는 일정
# 규모 이상"이란 영업장 면적 300㎡ 이상을 말한다. 이 미만이면 완제품만 판매해도
# 신고 대상이 아니다.
_ETC_FOOD_SALES_MIN_AREA_SQM = 300

# 판정 결과 → (근거 조문, 판정 사유) 매핑. permit 도메인의 _act_type_reason처럼
# "왜 이 판정인가"를 RAG가 아니라 규칙 자체가 근거로 들고 있게 한다(Rule
# Justification). 여기 없는 값(해당없음ᆞ신고대상제외)은 아래에서 따로 처리.
_FOOD_BUSINESS_RULE: dict[str, tuple[str, str]] = {
    "제과점영업": (
        "식품위생법 시행령 제21조제8호바목",
        "주로 빵ᆞ떡ᆞ과자를 제조ᆞ판매하는 영업",
    ),
    "일반음식점영업": (
        "식품위생법 시행령 제21조제8호나목",
        "음식류를 조리ᆞ판매하며 식사와 함께 음주가 허용되는 영업",
    ),
    "휴게음식점영업": (
        "식품위생법 시행령 제21조제8호가목",
        "음식류를 조리ᆞ판매하되 음주가 허용되지 않는 영업",
    ),
    "기타식품판매업": (
        "식품위생법 시행령 제21조제5호나목6)ᆞ시행규칙 제39조",
        "직접 제조ᆞ가공 없이 완제품을 판매하며 영업장 면적이 300㎡ 이상인 업소",
    ),
}

# 식품위생법상 별도 영업신고가 필요 없는 판정 결과(rag_query도 만들지 않는다).
NO_REPORT_RESULTS = ("해당없음", "신고대상제외")


def classify_food_business(facts: dict) -> str | None:
    """food_facts로 식품위생법 시행령 제21조상 영업 종류를 결정. 정보 부족하면 None.

    facts에서 쓰는 키:
      serves_food(bool): 음식ᆞ식품을 취급하는 업종인지. False면 식품위생법상
        영업신고 대상 자체가 아님(예: 사무실ᆞ미용실 등 비식품 업종).
      primarily_bakery(bool): 주로 빵ᆞ떡ᆞ과자를 제조ᆞ판매하는지 - 제21조8호바
        제과점영업 기준. 직접 제조ᆞ판매라 완제품 되팔기와는 다르다.
      manufactures_or_cooks(bool): 매장에서 직접 조리ᆞ제조ᆞ가공하는지. False면
        완제품만 판매하는 것이라 규모(store_area_sqm)로 기타식품판매업 여부가
        갈린다. True면 음식점(휴게/일반) 계열이다.
      store_area_sqm(float): 영업장 면적(㎡). 완제품만 판매할 때 기타식품판매업
        (300㎡ 이상, 시행규칙 제39조) 여부를 가른다.
      serves_alcohol(bool): 식사와 함께 음주행위(주류 판매)가 허용되는지 - 이게
        휴게음식점(불허, 제21조8호가)과 일반음식점(허용, 제21조8호나)을 가른다.

    모든 필드를 미리 요구하지 않고, 이미 확정된 사실만으로 결론이 나면 그
    자리에서 바로 판정한다(예: primarily_bakery=True만 알아도 제과점영업으로
    확정). 조리ᆞ제조하는 음식점 계열은 제과점일 가능성이 남아있으면(베이커리
    여부 미확정) 음식점으로 단정하지 않는다.
    """
    if facts.get("serves_food") is None:
        return None
    if not facts["serves_food"]:
        return "해당없음"  # 비식품 업종 - 영업신고 대상 아님

    if facts.get("primarily_bakery"):
        return "제과점영업"  # 시행령 제21조8호바

    manufactures = facts.get("manufactures_or_cooks")
    if manufactures is False:
        # 직접 조리ᆞ제조 없이 완제품만 판매 → 규모로 갈린다.
        area = facts.get("store_area_sqm")
        if area is None:
            return None
        if area >= _ETC_FOOD_SALES_MIN_AREA_SQM:
            return "기타식품판매업"  # 시행령 제21조5호나6)ᆞ시행규칙 제39조
        return "신고대상제외"  # 300㎡ 미만 완제품 판매 = 어느 종류에도 미해당

    # 여기부터는 조리ᆞ제조하는 음식점 계열. 베이커리 여부가 아직 미확정이면
    # 제과점일 가능성이 남아 음식점으로 단정하지 않는다.
    if facts.get("primarily_bakery") is None or manufactures is None:
        return None
    if facts.get("serves_alcohol") is None:
        return None
    return "일반음식점영업" if facts["serves_alcohol"] else "휴게음식점영업"  # 제21조8호나ᆞ가


def food_business_message(business_type: str) -> str:
    """classify_food_business()의 판정 결과를 LLM이 답변에 반영할 안내 문구로
    변환. procedure_stage_message(permit.py)와 동일한 역할 - 판정 사유ᆞ근거
    조문(Rule Justification)을 규칙 자체가 만들어 붙인다(RAG가 아니라).
    """
    if business_type == "해당없음":
        return "식품위생법상 별도 영업신고 대상이 아닙니다(음식류를 조리ᆞ판매하지 않는 업종)."
    if business_type == "신고대상제외":
        return (
            "식품위생법상 별도 영업신고 대상이 아닙니다 - 직접 제조ᆞ가공 없이 완제품만 "
            "판매하고 영업장 면적이 300㎡ 미만이면 어느 영업 종류에도 해당하지 않습니다"
            "(기타식품판매업은 영업장 300㎡ 이상만 신고 대상 - 식품위생법 시행규칙 제39조). "
            "다만 완제품이라도 별도 관리가 필요한 품목이 있을 수 있으니 관할 보건소에 최종 확인하세요."
        )
    article, reason = _FOOD_BUSINESS_RULE[business_type]
    return (
        f"식품위생법상 영업 종류 판정: {business_type} (사유: {reason} - {article}). "
        f"관할 보건소에 영업신고가 필요합니다."
    )


@tool
def record_food_facts(
    serves_food: bool | None = None,
    manufactures_or_cooks: bool | None = None,
    primarily_bakery: bool | None = None,
    serves_alcohol: bool | None = None,
    store_area_sqm: float | None = None,
) -> str:
    """식품위생법상 영업 종류 판정에 필요한 사용자 업종 정보를 알게 되는 대로 기록하세요.

    - 이번 턴에 새로 알게 된 필드만 채우고 나머지는 생략(None)하세요. 여러 턴에
      걸쳐 나눠서 호출해도 이전에 기록한 값 위에 누적됩니다.
    - 영업 종류(휴게음식점/일반음식점 등) 판정은 여기서 계산 안 합니다 - 정보가
      충분히 모이면 시스템이 자동으로(규칙 기반) 판정하니, 사용자에게 "휴게음식점
      대상인가요 일반음식점 대상인가요?" 같은 질문은 하지 마세요. 사용자가 실제로
      답할 수 있는 사실(직접 조리ᆞ제조 여부ᆞ주류 판매 여부 등)만 물어보고 기록하세요.
    - 카페ᆞ식당처럼 음식을 다루는 업종이면 serves_food=True를 먼저 기록하고,
      사무실ᆞ미용실처럼 식품위생법과 무관한 업종이면 serves_food=False만 기록해도
      곧바로 "해당없음"으로 판정됩니다.
    - **직접 조리ᆞ제조 없이 이미 완성된 완제품만 되파는 경우**(예: 캔 음료ᆞ포장
      과자만 진열 판매)는 manufactures_or_cooks=False로 기록하세요. 이 경우
      영업장 면적(store_area_sqm)이 판정을 가릅니다(300㎡ 이상이면 기타식품판매업
      신고 대상, 미만이면 신고 대상 아님) - 면적도 함께 물어보세요.

    Args:
        serves_food: 음식ᆞ식품을 취급하는 업종인지 여부
        manufactures_or_cooks: 매장에서 직접 조리ᆞ제조ᆞ가공하는지(완제품만 되팔면 False)
        primarily_bakery: 주로 빵ᆞ떡ᆞ과자를 제조ᆞ판매하는지
        serves_alcohol: 식사와 함께 음주행위(주류 판매)가 허용되는지
        store_area_sqm: 영업장 면적(㎡) - 완제품만 판매할 때 기타식품판매업 판정용
    """
    logger.info("[도구] record_food_facts(%r)", {
        k: v for k, v in locals().items() if v is not None
    })
    return "기록됨."
