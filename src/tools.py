"""에이전트가 사용할 도구 정의.

@tool 데코레이터로 정의된 함수는 LLM이 자동으로 호출 가능.
docstring이 LLM에게 도구 선택 힌트 - 명확하게 작성 필수.
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

from src import config
from src import parent_store
from src.retriever import keyword_search, search_with_score

logger = logging.getLogger(__name__)


# 카테고리별 한글 이름 매핑
CATEGORY_NAMES = {
    "laws": "법령",
    "ordinances": "서울시 조례",
    "procedures": "절차 안내",
}


@tool
def search_regulations(query:str) -> str:
    """건축 인허가 관련 법령/조례/절차 문서를 검색합니다.
    
    다음과 같은 경우에 사용하세요:
    - 건축법, 시행령, 시행규칙 관련 조항
    - 서울시 건축조례, 도시계획조례
    - 건축신고, 건축허가, 용도변경 절차
    - 근린생활시설(카페, 사무실 등) 관련 규정
    - 주차, 정화조, 소방 등 부대시설 요구사항
    - 필요 서류 및 처리 기간
    
    Args:
        query: 검색할 키워드나 질문 (한국어)
    
    Returns:
        관련 법령/조례 조항의 요약 텍스트.
    """
    logger.info("[도구] search_regulations(query=%r)", query)

    results = search_with_score(query, k=config.TOP_K)

    if not results:
        return "관련 규정를 찾을 수 없습니다."

    # 부모-자식 청킹: 검색은 작은 자식 청크로 하되, LLM에는 그 조항 전체를 보여줌.
    # 같은 조항의 자식 청크 여러 개가 매칭되면 한 번만 포함(중복 제거).
    seen_articles: set[tuple[str, str]] = set()
    parts = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        article_id = doc.metadata.get("article_id")

        if article_id is not None:
            key = (source, article_id)
            if key in seen_articles:
                continue
            seen_articles.add(key)

        category = doc.metadata.get("category", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")  # 두번째는 디폴트값
        similarity = 1 - score
        category_kr = CATEGORY_NAMES.get(category, category)

        content = doc.page_content
        if article_id is not None:
            parent_content = parent_store.get_parent(source, article_id)
            if parent_content is not None:
                content = parent_content

        parts.append(
            f"[문서 {len(parts) + 1} | {category_kr}, 청크: {chunk_idx}, 유사도: {similarity:.3f}]\n"
            f"{content}"
        )

    return "\n\n".join(parts)


@tool
def search_by_term(keyword: str) -> str:
    """법률 용어의 정의/뜻을 물어보는 질문에 사용하는 키워드(정확 매칭) 검색 도구.

    예: "건폐율이 뭐야", "용적률의 정의는?", "이격거리란 무엇인가요"
    -> 이런 질문에서는 핵심 법률 용어(예: "건폐율")만 추출해서 이 도구에 전달하세요.

    search_regulations(유사도 검색)는 조문이 딱딱한 문어체라 "~란 무엇인가요"
    같은 자연어 질문과 의미 유사도가 낮게 나와 정의 조항을 놓칠 수 있습니다.
    이 도구는 용어를 포함한 조항을 부분 문자열로 정확히 찾아냅니다.

    Args:
        keyword: 찾고자 하는 핵심 법률 용어 (예: "건폐율", "용적률", "대지면적")

    Returns:
        해당 용어가 포함된 조항들의 텍스트 (정의 조항 우선 정렬).
    """
    logger.info("[도구] search_by_term(keyword=%r)", keyword)

    docs = keyword_search(keyword)

    if not docs:
        return f"'{keyword}'를 포함한 조항을 찾을 수 없습니다."

    parts = []
    for i, doc in enumerate(docs, 1):
        category = doc.metadata.get("category", "unknown")
        article_id = doc.metadata.get("article_id", "?")
        category_kr = CATEGORY_NAMES.get(category, category)

        parts.append(
            f"[문서 {i} | {category_kr}, 조항: 제{article_id}조]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(parts)


TOOLS = [search_regulations, search_by_term]
