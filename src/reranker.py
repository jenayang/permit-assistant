"""검색 결과 재정렬(CrossEncoder) + 준-중복 제거.

retriever.py의 벡터 유사도 검색은 임베딩 하나로 폭넓게 후보를 뽑는 데는
강하지만 순위 정밀도는 떨어진다. 여기서는 (query, 문서) 쌍을 직접 채점하는
CrossEncoder로 후보를 다시 정렬해서 상위 몇 개만 추린다 - sentence-transformers가
이미 프로젝트 의존성에 있어 새 라이브러리 없이 쓸 수 있다.
"""
from __future__ import annotations

import difflib

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# 다국어(한국어 포함) 지원 CrossEncoder. 임베딩 모델(multilingual-e5-base)처럼
# 최초 호출 시 HuggingFace에서 자동 다운로드된다. 학습 시점 지식 기반 선택이라
# 실제 한국어 품질은 첫 사용 시 스모크 테스트로 확인할 것.
_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(_MODEL_NAME)
    return _cross_encoder


def rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """CrossEncoder 점수 기준으로 docs를 재정렬해서 top_k만 반환."""
    if not docs:
        return docs
    pairs = [(query, d.page_content) for d in docs]
    scores = _get_cross_encoder().predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def dedup_near_duplicates(docs: list[Document], threshold: float = 0.85) -> list[Document]:
    """법ᆞ시행령ᆞ시행규칙ᆞ조례가 같은 내용을 반복 서술하는 경우, 이미 채택된
    문서와 텍스트가 threshold 이상 겹치면 스킵한다(stdlib difflib, 새 의존성 없음).

    조항 단위 동일성(같은 (source, article_id))은 regulation.py에서 이미
    별도로 걸러지므로, 여기서는 "다른 조항인데 내용이 사실상 같은" 경우만 대상.
    """
    kept: list[Document] = []
    for doc in docs:
        is_dup = any(
            difflib.SequenceMatcher(None, doc.page_content, k.page_content).ratio() > threshold
            for k in kept
        )
        if not is_dup:
            kept.append(doc)
    return kept
