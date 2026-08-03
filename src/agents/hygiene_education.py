"""Hygiene Education Agent - 위생교육 안내(RAG 기반).

원래는 식품위생법 시행규칙을 원문 대조해 하드코딩한 정적 안내였다. 2026-07-28
검증(RAG vs 정적 콘텐츠 비교)에서 RAG가 같은 결론(사전교육 6시간, 시행령
제21조제8호, 원격교육 가능)에 정확히 도달하는 걸 확인해서 하드코딩 대신
매번 search_regulations로 최신 법령을 직접 찾아 반환하도록 전환했다.

실제 교육을 운영하는 기관명(한국외식업중앙회 등)은 법조문에 안 나오고
협회 자체 정보라 정부 공식 출처만큼 신뢰도가 높지 않다 - 검색 결과에도
당연히 안 나오니, 도구 사용 지침에서 "모르면 관할 보건소 확인 권장"으로
안내하도록 남겨뒀다(RAG로 바뀌어도 이 원칙은 그대로 적용).
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

from src.rag.regulation import search_regulations

logger = logging.getLogger(__name__)

_QUERIES = [
    "위생교육은 몇 시간 받아야 하고 언제까지 받아야 하나요?",
    "위생교육을 온라인으로도 받을 수 있나요?",
]


@tool
def get_hygiene_education_guide() -> str:
    """식품위생교육(위생교육) 대상ᆞ교육시간ᆞ이수시기ᆞ교육방법을 법령에서 검색해 반환합니다.

    - 위생교육을 언제ᆞ몇 시간ᆞ어떻게 받아야 하는지 물으면 이 도구를
      호출하세요(내부적으로 식품위생법 시행규칙을 여러 관점에서 검색해
      합쳐서 돌려줍니다 - 직접 다시 search_regulations를 호출할 필요
      없습니다).
    - 실제 교육 운영기관ᆞ교육비처럼 검색 결과에 없는 세부사항은 지어내지
      말고 "관할 보건소에 확인하라"고 안내하세요 - 임의로 가격이나 기관을
      단정해서 덧붙이지 마세요.
    """
    logger.info("[도구] get_hygiene_education_guide()")
    return "\n\n".join(search_regulations.func(q) for q in _QUERIES)
