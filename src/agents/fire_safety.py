"""Fire Safety Agent - 카페ᆞ음식점(근린생활시설)에 필요한 소방시설 판정(규칙 기반).

permit.py/food_safety.py와 동일한 원칙: 어떤 소방시설을 설치해야 하는지는
소방시설 설치 및 관리에 관한 법률 시행령이 정한 객관적 기준(연면적ᆞ업종 등)
으로 정해지므로 LLM 판단에 맡기지 않고 classify_fire_safety()가 파이썬
규칙으로 결정론적으로 계산한다(원문 대조: data/laws/소방시설 설치 및 관리에
관한 법률 시행령(대통령령)(제36432호)(20260701).pdf, 별표2ᆞ별표4).

원문 확인 결과, 휴게음식점ᆞ제과점ᆞ일반음식점은 면적과 무관하게 전부
"근린생활시설"(별표2 2호나목)에 해당한다 - 그래서 이 챗봇의 타깃(카페ᆞ음식점
창업)에서는 "근린생활시설 해당 여부"를 별도로 물어볼 필요 없이 바로 연면적
기준 판정으로 들어간다.

permit.py의 classify_case/food_safety.py의 classify_food_business와 달리
이 도메인은 결과가 "단 하나"가 아니다 - 규모가 커지면 여러 소방시설이 동시에
필요해질 수 있으므로(예: 연면적 700㎡면 소화기구ᆞ비상경보설비ᆞ자동화재탐지설비
셋 다 해당) 결과를 리스트로 반환한다. 빈 리스트(`[]`)는 "필요한 소방시설
없음"이라는 유효한 결과이지 미분류(`None`)가 아니다 - 호출부는 반드시 이
둘을 구분해서 다뤄야 한다.
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

from src.agents.legal_data import get_thresholds

logger = logging.getLogger(__name__)


def classify_fire_safety(facts: dict) -> list[str] | None:
    """fire_facts로 필요한 소방시설 목록을 결정. 정보 부족하면 None.

    facts에서 쓰는 키:
      size_sqm: 연면적(㎡) - 소화기구ᆞ비상경보설비ᆞ자동화재탐지설비ᆞ
        간이스프링클러설비 여부를 가르는 기준(전부 시행령 별표4의 순수
        면적 임계값 - 실제 숫자는 permit_thresholds.yaml의
        소방시설_연면적기준_sqm에서 로드. permit.py와 같은 이유로 임계값만
        분리 - 법령 개정으로 숫자만 바뀌면 이 파일은 안 건드리고 YAML만
        고치면 됨).
      is_large_store_tenant: 유통산업발전법상 대규모점포(백화점ᆞ쇼핑센터
        등)에 입점한 일반음식점인지 여부 - 상업용 주방자동소화장치 대상
        기준(별표4 1.나.2.가). 독립 점포로 창업하는 대부분의 카페ᆞ음식점은
        False에 해당하지만, 별도로 확인 없이 넘겨짚지 않는다 - 처음엔
        "카페ᆞ음식점은 대부분 해당"으로 짐작했다가 원문 대조 후(2026-07-27)
        실제로는 대규모점포 입점ᆞ집단급식소 두 경우로 한정된다는 걸 확인한
        사례라, 이 필드를 생략하지 않고 명시적으로 받는다.
    """
    if facts.get("size_sqm") is None or facts.get("is_large_store_tenant") is None:
        return None

    size = facts["size_sqm"]
    thresholds = get_thresholds()["소방시설_연면적기준_sqm"]
    required: list[str] = []
    if size >= thresholds["소화기구"]:
        required.append("소화기구")  # 별표4 1.가.1)
    if size >= thresholds["비상경보설비"]:
        required.append("비상경보설비")  # 별표4 2.나.1)
    if size >= thresholds["자동화재탐지설비"]:
        required.append("자동화재탐지설비")  # 별표4 2.다.3) 근린생활시설(목욕장 제외) 기준
    if size >= thresholds["간이스프링클러설비"]:
        required.append("간이스프링클러설비")  # 별표4 1.마.2)가) 근린생활시설 바닥면적 합계 기준
    if facts["is_large_store_tenant"]:
        required.append("상업용 주방자동소화장치")  # 별표4 1.나.2)가) 대규모점포 입점 일반음식점(임계값 없음 - 여부만 판단)

    return required


def fire_safety_message(required: list[str]) -> str:
    """classify_fire_safety()의 판정 결과를 LLM이 답변에 반영할 안내 문구로
    변환. procedure_stage_message(permit.py)ᆞfood_business_message(food_safety.py)
    와 동일한 역할.
    """
    if not required:
        return "현재 규모 기준으로는 별도로 설치해야 하는 소방시설이 없습니다(연면적 33㎡ 미만 등)."
    return f"연면적 등 기준으로 설치해야 하는 소방시설: {', '.join(required)}."


@tool
def record_fire_facts(
    size_sqm: float | None = None,
    is_large_store_tenant: bool | None = None,
) -> str:
    """소방시설 설치 대상 판정에 필요한 사실을 알게 되는 대로 기록하세요.

    - 이번 턴에 새로 알게 된 필드만 채우고 나머지는 생략(None)하세요. 여러 턴에
      걸쳐 나눠서 호출해도 이전에 기록한 값 위에 누적됩니다.
    - size_sqm은 case_facts(record_case_facts)의 size_sqm과 별개로 여기도
      기록해야 합니다 - 도메인마다 상태가 독립적입니다. 이미 건축 인허가
      판정용으로 연면적을 들었다면 같은 값을 그대로 여기도 기록하세요.
    - 소방시설 종류 자체는 여기서 계산 안 합니다 - 정보가 충분히 모이면
      시스템이 자동으로(규칙 기반) 판정하니, 사용자에게 "소화기구가 필요한가요?"
      같은 질문은 하지 마세요. 사용자가 실제로 답할 수 있는 사실만 물어보세요.
    - is_large_store_tenant: 백화점ᆞ쇼핑센터 등 대규모점포에 입점하는 형태가
      아니라 독립 점포로 창업하는 경우가 명백하면 False로 기록하세요.

    Args:
        size_sqm: 연면적(㎡)
        is_large_store_tenant: 유통산업발전법상 대규모점포에 입점한 일반음식점인지 여부
    """
    logger.info("[도구] record_fire_facts(%r)", {
        k: v for k, v in locals().items() if v is not None
    })
    return "기록됨."
