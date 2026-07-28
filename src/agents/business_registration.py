"""Business Registration Agent - 사업자등록 안내(RAG 기반).

원래는 국세청 홈택스 공식 페이지를 원문 대조해 하드코딩한 정적 안내였다.
2026-07-28 검증(RAG vs 정적 콘텐츠 비교) 전에는 "부가가치세법ᆞ시행령이
data/laws에 없어서 RAG가 답을 못 찾는다"는 게 정적 콘텐츠를 쓴 이유였는데,
그 법령을 실제로 추가ᆞ재인덱싱한 뒤 재검증하니 RAG가 정적 콘텐츠와 동일한
결론(20일 이내 신청, 제출서류 등)에 정확히 도달했다 - 그래서 하드코딩 대신
매번 search_regulations로 최신 법령을 직접 찾아 반환하도록 전환했다. 법이
개정돼도 재인덱싱만 하면 자동 반영되고, 원문 대조를 다시 안 해도 된다는 게
정적 콘텐츠 대비 장점이다.

고정 질의 3개(신청 기한ᆞ방법 / 필요 서류 / 간이과세 기준)로 검색하는 이유:
"사업자등록 어떻게 해요?"처럼 넓은 질문 하나로 한 번만 검색하면 벡터 유사도가
분산돼 필요한 조항을 다 못 찾는 경우가 있었다(4대보험 검증 때도 동일 문제
확인) - 하위 주제별로 나눠 검색해서 합치는 편이 훨씬 안정적이다.
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

from src.agents.regulation import search_regulations

logger = logging.getLogger(__name__)

_QUERIES = [
    "사업자등록은 언제까지, 어떻게 해야 하나요?",
    "사업자등록에 필요한 서류가 뭔가요?",
    "간이과세자 기준 공급대가",
]


@tool
def get_business_registration_guide() -> str:
    """사업자등록 신청 절차ᆞ필요서류ᆞ과세유형(간이/일반)을 법령에서 검색해 반환합니다.

    - 사업자등록을 어떻게 하는지, 언제까지 해야 하는지, 무슨 서류가
      필요한지 물으면 이 도구를 호출하세요(내부적으로 부가가치세법ᆞ
      시행령을 여러 관점에서 검색해 합쳐서 돌려줍니다 - 직접 다시
      search_regulations를 호출할 필요 없습니다).
    - **이미 식품위생법 영업신고 판정(food_result)이 나온 사용자라면,
      검색 결과의 "허가ᆞ등록ᆞ신고증 사본"이 바로 그 영업신고증이라는 걸
      답변에서 연결해서 설명하세요** - 로드맵 단계끼리 이어지는 느낌을
      주는 지점입니다.
    - 간이과세자/일반과세자 중 어느 쪽에 해당하는지 세부 판단은 검색 결과
      범위 밖입니다 - 기준 금액만 안내하고, 정확한 선택은 세무서ᆞ세무사
      상담을 권하세요. 사용자 매출을 추정해서 단정하지 마세요.
    """
    logger.info("[도구] get_business_registration_guide()")
    return "\n\n".join(search_regulations.func(q) for q in _QUERIES)
