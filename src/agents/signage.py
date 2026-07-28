"""Signage Agent - 간판ᆞ옥외광고물 허가/신고 판정(규칙 기반).

permit.py의 classify_case와 동일한 원칙ᆞ패턴(act_type 분기 → 결정론적 판정)을
그대로 재사용한다 - sign_type별로 필요한 사실이 다르므로 REQUIRED_FIELDS_SIGNAGE로
분기별 필수 필드를 관리한다.

원문 대조 완료(2026-07-27):
- data/laws/옥외광고물 등의 관리와 옥외광고산업 진흥에 관한 법률 시행령
  제4조(허가 대상)ᆞ제5조(신고 대상)
- data/ordinances/서울특별시 옥외광고물 등의 관리와 옥외광고산업 진흥에
  관한 조례(입간판 표시방법 등 세부 규격 - 제9조의2)

건축 조례(서울시 전체 단일)와 달리, 옥외광고물 조례는 원래 자치구별로
따로 있다(강남구 조례를 먼저 받았다가 서울시 전체 조례로 다시 교체한
이력 있음, 2026-07-27) - 지금 반영한 규칙은 법률ᆞ시행령(전국 공통)과
서울시 조례 수준까지고, 자치구별 세부 표시방법 차이(예: 색상ᆞ재질 제한)는
이 판정 범위 밖이다. 허가/신고 여부 자체는 법률ᆞ시행령이 결정하므로 이
판정 결과는 자치구와 무관하게 정확하다.

카페ᆞ음식점 창업자가 실제로 마주치는 5종류만 다룬다(교통수단 이용
광고물ᆞ선전탑ᆞ아치광고물 등 창업과 무관한 종류는 범위 밖).
"""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# sign_type별로 판정에 필요한 사실 목록. 하나라도 비어있으면(None) 아직
# 분류 안 함 - permit.py의 REQUIRED_FIELDS와 동일한 역할.
REQUIRED_FIELDS_SIGNAGE: dict[str, list[str]] = {
    "벽면이용간판": ["length_m", "floor", "is_third_party_ad", "area_sqm"],
    "돌출간판": ["height_m", "area_sqm", "is_medical_or_salon_sign"],
    "지주이용간판": ["height_m"],
    "입간판": [],  # 시행령 제5조1항5의2호에 따라 항상 신고 대상 - 물을 사실 없음
    "현수막": ["display_facility_area_sqm"],
}


def _missing_fields_signage(sign_type: str, facts: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS_SIGNAGE.get(sign_type, []) if facts.get(f) is None]


def classify_signage(facts: dict) -> str | None:
    """signage_facts로 허가/신고/불필요 여부를 결정. 정보 부족하면 None.

    facts에서 쓰는 키:
      sign_type: "벽면이용간판"ᆞ"돌출간판"ᆞ"지주이용간판"ᆞ"입간판"ᆞ"현수막" 중 하나
      length_m: (벽면이용간판) 간판 한 변의 길이(m)
      floor: (벽면이용간판) 설치하려는 층수
      is_third_party_ad: (벽면이용간판) 자기 업소가 아닌 광고(타사광고)인지 여부
      area_sqm: (벽면이용간판ᆞ돌출간판) 간판 면적(㎡)
      height_m: (돌출간판ᆞ지주이용간판) 지면으로부터 윗부분까지 높이(m)
      is_medical_or_salon_sign: (돌출간판) 의료기관ᆞ약국ᆞ이용업소ᆞ미용업소의
        표지등인지 여부(해당하면 허가 대상에서 제외되어 신고로 완화됨)
      display_facility_area_sqm: (현수막) 현수막을 설치하는 게시시설 자체의 면적(㎡)

    간이 처리 - "면적 5㎡ 이상"의 예외("건물 출입구 양옆에 세로로 표시하는
    것은 제외")는 이 판정에 반영하지 않았다(시행령 제5조1항1호가목 단서) -
    실제로 이 예외에 해당하는 극소수 케이스는 관할 구청 확인을 안내하는 걸로
    갈음한다.
    """
    sign_type = facts.get("sign_type")
    if sign_type is None:
        return None

    # 벽면이용간판ᆞ돌출간판은 2026-07-28 전까지 진입 전에 REQUIRED_FIELDS_SIGNAGE
    # 전체를 무조건 요구했다 - 그런데 예를 들어 length_m=15(10m 이상) 하나만
    # 알아도 이미 "허가"가 확정되는데(아래 주석 참고), floor/is_third_party_ad/
    # area_sqm까지 다 모여야만 판정을 내려서 사용자가 이미 답이 나온 값을
    # 계속 추가로 질문받는 문제가 있었다(docs/test_scenarios.md류 경계값
    # 점검 중 permit.py의 대수선에서 같은 패턴을 먼저 발견ᆞ수정한 뒤 이
    # 파일도 감사해서 발견). 아래 두 분기는 부분 정보로도 안전하게 확정
    # 가능한 조건만 먼저 확인하고, 그렇지 않으면 나머지 필드를 마저 요구한다.
    # "허가"(더 엄격한 쪽) 방향으로만 조기 확정을 허용한다는 원칙을 지킨다 -
    # 신고ᆞ불필요(더 완화된 쪽)를 부분 정보로 성급히 말했다가 나중에 실제로는
    # 허가 대상이었던 걸로 드러나는 위험을 피하기 위함.

    if sign_type == "벽면이용간판":
        # 시행령 제4조1항1호: 한 변 10m 이상 → 다른 필드와 무관하게 즉시 허가.
        if facts.get("length_m") is not None and facts["length_m"] >= 10:
            return "허가"
        # 시행령 제4조1항1호: 4층 이상 + 타사광고 → 허가(두 필드 다 알아야
        # 확정/배제 가능 - floor만 알고 is_third_party_ad를 모르면 신고인지
        # 허가인지 아직 갈릴 수 있다).
        if (
            facts.get("floor") is not None and facts.get("is_third_party_ad") is not None
            and facts["floor"] >= 4 and facts["is_third_party_ad"]
        ):
            return "허가"
        # 위 두 "허가" 조건을 완전히 배제해야 신고ᆞ불필요를 안전하게 말할 수
        # 있는데, 그러려면 결국 네 필드가 다 필요하다.
        if any(facts.get(f) is None for f in ("length_m", "floor", "is_third_party_ad", "area_sqm")):
            return None
        # 시행령 제5조1항1호: 면적 5㎡ 이상, 또는 4층 이상(자사광고) → 신고
        if facts["area_sqm"] >= 5:
            return "신고"
        if facts["floor"] >= 4:
            return "신고"
        return "허가ᆞ신고 불필요"

    if sign_type == "돌출간판":
        # 시행령 제4조1항2호: 원칙 허가. 다만 의료ᆞ약국ᆞ이용ᆞ미용업소
        # 표지등이거나, 높이 5m 미만이거나, 면적 1㎡ 미만이면(셋 중 하나만
        # 해당돼도, OR 조건) 허가 대상에서 제외되어 신고로 완화(제5조1항4호).
        # OR 조건이라 셋 중 하나라도 먼저 확인되면 나머지를 몰라도 즉시 신고로
        # 확정할 수 있다(다른 값이 뭐든 이 조건 하나만으로 이미 완화 대상).
        if facts.get("is_medical_or_salon_sign"):
            return "신고"
        if facts.get("height_m") is not None and facts["height_m"] < 5:
            return "신고"
        if facts.get("area_sqm") is not None and facts["area_sqm"] < 1:
            return "신고"
        # 셋 다 완화 조건에 안 걸리는 걸 확인해야만(=셋 다 알아야만) 허가로
        # 확정할 수 있다.
        if any(facts.get(f) is None for f in ("height_m", "area_sqm", "is_medical_or_salon_sign")):
            return None
        return "허가"

    if sign_type == "지주이용간판":
        if _missing_fields_signage(sign_type, facts):
            return None
        # 시행령 제4조1항5호/제5조1항5호: 높이 4m 이상 허가, 미만 신고
        return "허가" if facts["height_m"] >= 4 else "신고"

    if sign_type == "입간판":
        # 시행령 제5조1항5의2호: 종류 자체로 항상 신고 대상(규격은 서울시
        # 조례 제9조의2가 별도로 정함 - 높이 1.2m 이하ᆞ면적 0.6㎡ 이하 등)
        return "신고"

    if sign_type == "현수막":
        if _missing_fields_signage(sign_type, facts):
            return None
        # 시행령 제4조2항2호/제5조2항: 게시시설 면적 30㎡ 초과면 허가, 그 외 신고
        return "허가" if facts["display_facility_area_sqm"] > 30 else "신고"

    return None


def signage_message(sign_type: str, result: str) -> str:
    """classify_signage()의 판정 결과를 LLM이 답변에 반영할 안내 문구로 변환.
    procedure_stage_message(permit.py) 등과 동일한 역할."""
    if result == "허가ᆞ신고 불필요":
        return f"{sign_type}은(는) 현재 규격 기준으로 별도 허가ᆞ신고 없이 설치할 수 있습니다."
    return f"{sign_type} 판정: {result} 대상입니다. 관할 구청(옥외광고물 담당 부서)에 {result} 절차를 진행하세요."


@tool
def record_signage_facts(
    sign_type: Literal["벽면이용간판", "돌출간판", "지주이용간판", "입간판", "현수막"] | None = None,
    length_m: float | None = None,
    floor: int | None = None,
    is_third_party_ad: bool | None = None,
    area_sqm: float | None = None,
    height_m: float | None = None,
    is_medical_or_salon_sign: bool | None = None,
    display_facility_area_sqm: float | None = None,
) -> str:
    """간판ᆞ옥외광고물 허가/신고 판정에 필요한 사실을 알게 되는 대로 기록하세요.

    - 이번 턴에 새로 알게 된 필드만 채우고 나머지는 생략(None)하세요. 여러 턴에
      걸쳐 나눠서 호출해도 이전에 기록한 값 위에 누적됩니다.
    - sign_type부터 먼저 파악하세요 - 어떤 종류의 간판인지에 따라 필요한
      나머지 필드가 달라집니다(예: 입간판은 sign_type 하나만 알면 판정
      끝남 - 항상 신고 대상).
    - 허가/신고/불필요 여부 자체는 여기서 계산 안 합니다 - 정보가 충분히
      모이면 시스템이 자동으로(규칙 기반) 판정하니, 사용자에게 "이거 허가
      대상인가요?"라고 묻지 마세요. 사용자가 실제로 답할 수 있는 사실
      (크기ᆞ높이ᆞ층수ᆞ광고 대상 등)만 물어보세요.
    - is_third_party_ad(타사광고)는 일반인이 바로 이해 못 할 수 있는 용어이니
      "본인 업소 광고인가요, 아니면 다른 업체 광고를 걸어주는 건가요?"처럼
      풀어서 물어보세요.

    Args:
        sign_type: 간판 종류(벽면이용간판/돌출간판/지주이용간판/입간판/현수막)
        length_m: (벽면이용간판) 간판 한 변의 길이(m)
        floor: (벽면이용간판) 설치하려는 층수
        is_third_party_ad: (벽면이용간판) 타사광고 여부(자기 업소 광고면 False)
        area_sqm: (벽면이용간판ᆞ돌출간판) 간판 면적(㎡)
        height_m: (돌출간판ᆞ지주이용간판) 지면으로부터 윗부분까지 높이(m)
        is_medical_or_salon_sign: (돌출간판) 의료기관ᆞ약국ᆞ이용ᆞ미용업소 표지등 여부
        display_facility_area_sqm: (현수막) 게시시설 자체의 면적(㎡)
    """
    logger.info("[도구] record_signage_facts(%r)", {
        k: v for k, v in locals().items() if v is not None
    })
    return "기록됨."
