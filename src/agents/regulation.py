"""Regulation Agent - 법령/조례/절차 문서 검색(RAG).

판단은 하지 않고 검색과 근거 제공만 담당한다(설계서 5번 섹션 참고).
실제 판정은 permit.classify_case()가, 최종 설명 생성은 agent.py의 LLM이 담당.
"""
from __future__ import annotations

import logging
import re
import unicodedata

from langchain_core.tools import tool

from src import config
from src import parent_store
from src.reranker import dedup_near_duplicates, rerank
from src.retriever import keyword_search, search_with_score

logger = logging.getLogger(__name__)


# 카테고리별 한글 이름 매핑
CATEGORY_NAMES = {
    "laws": "법령",
    "ordinances": "서울시 조례",
    "procedures": "절차 안내",
}

# 법령 계층(법/시행령/시행규칙/조례) 태그. data/laws, data/ordinances의 파일명이
# 이미 "건축법(법률)(제21035호)(20260227).pdf"처럼 괄호로 계층을 포함하고
# 있어서, 인제스천(law_chunker.py) 수정 없이 파일명 파싱만으로 뽑아낼 수 있다.
# "법령 충돌 검사"를 자동 탐지가 아니라 이 태그 + 프롬프트 우선순위 안내로
# 스코프를 좁힌 것(자동 충돌 탐지는 범위 밖 - agent.py SYSTEM_PROMPT 참고).
_TIER_PATTERN = re.compile(r"\((법률|대통령령|총리령|국토교통부령|환경부고시|[가-힣]+조례)\)")
_TIER_LABELS = {
    "법률": "법", "대통령령": "시행령", "총리령": "시행규칙",
    "국토교통부령": "시행규칙", "환경부고시": "고시",
}


def _law_tier(source: str) -> str:
    # macOS(APFS)가 일부 파일을 유니코드 NFD(자모 분해형)로 저장해서, 같은
    # 글자처럼 보여도 source 문자열이 바이트 단위로 달라 정규식이 안 맞는
    # 경우가 있다(실제로 겪음) - NFC로 정규화한 뒤 매칭한다.
    match = _TIER_PATTERN.search(unicodedata.normalize("NFC", source))
    if not match:
        return "기타"
    tag = match.group(1)
    return "조례" if tag.endswith("조례") else _TIER_LABELS.get(tag, "기타")


@tool
def search_regulations(query: str) -> str:
    """건축 인허가 관련 법령/조례/절차 문서를 검색합니다.

    다음과 같은 경우에 사용하세요:
    - 건축법, 시행령, 시행규칙 관련 조항
    - 서울시 건축조례, 도시계획조례
    - 건축신고, 건축허가, 용도변경 절차
    - 건축물 용도(근린생활시설ᆞ주거ᆞ업무ᆞ산업시설 등) 관련 규정
    - 주차, 정화조, 소방 등 부대시설 요구사항
    - 필요 서류 및 처리 기간

    Args:
        query: 검색할 키워드나 질문 (한국어)

    Returns:
        관련 법령/조례 조항의 요약 텍스트.
    """
    logger.info("[도구] search_regulations(query=%r)", query)

    # 1차로 넓게 후보를 뽑은 뒤(RERANK_CANDIDATE_K) CrossEncoder로 재정렬해서
    # 상위 RERANK_TOP_K만 남기고, 준-중복(다른 조항인데 내용이 사실상 같은 경우)을
    # 걸러낸다. 원래 벡터 유사도 점수는 재정렬 후 순서가 바뀌므로 더 안 쓴다.
    results = search_with_score(query, k=config.RERANK_CANDIDATE_K)
    if not results:
        return "관련 규정를 찾을 수 없습니다."

    candidates = [doc for doc, _ in results]
    reranked = rerank(query, candidates, config.RERANK_TOP_K)
    reranked = dedup_near_duplicates(reranked)

    # 부모-자식 청킹: 검색은 작은 자식 청크로 하되, LLM에는 그 조항 전체를 보여줌.
    # 같은 조항의 자식 청크 여러 개가 매칭되면 한 번만 포함(중복 제거).
    seen_articles: set[tuple[str, str]] = set()
    parts = []
    for doc in reranked:
        source = doc.metadata.get("source", "unknown")
        article_id = doc.metadata.get("article_id")

        if article_id is not None:
            key = (source, article_id)
            if key in seen_articles:
                continue
            seen_articles.add(key)

        category = doc.metadata.get("category", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")  # 두번째는 디폴트값
        category_kr = CATEGORY_NAMES.get(category, category)
        tier = _law_tier(source)

        content = doc.page_content
        if article_id is not None:
            parent_content = parent_store.get_parent(source, article_id)
            if parent_content is not None:
                content = parent_content

        parts.append(
            f"[문서 {len(parts) + 1} | {category_kr} · {tier}, 청크: {chunk_idx}]\n"
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
        tier = _law_tier(doc.metadata.get("source", ""))

        parts.append(
            f"[문서 {i} | {category_kr} · {tier}, 조항: 제{article_id}조]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(parts)
