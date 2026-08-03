"""RAG(검색-증강생성) 파이프라인 패키지 - retriever/reranker/ingestion/law_chunker/
parent_store/regulation을 한곳에 묶는다. 원래 src/ 최상위에 흩어져 있던 파일들인데,
서로를 실제로 호출하며 하나의 검색 파이프라인을 이루므로 이 폴더로 옮겼다.

이 폴더로 옮기며 모듈 경로가 바뀌었다(예: `src.retriever` → `src.rag.retriever`,
`src.agents.regulation` → `src.rag.regulation`) - 그 경로를 쓰던 agent.py/api.py/
pipeline.py/classify/pipeline.py 등의 import를 함께 갱신했다.

law_chunker는 ingestion.py 내부에서만 쓰여 여기서 재노출하지 않는다.
`parent_store`/`retriever`/`ingestion`은 모듈 자체를 참조하는 기존 코드
(`from src import parent_store` 등)가 있어 서브모듈도 함께 노출한다.
"""
from src.rag import ingestion, parent_store, retriever
from src.rag.ingestion import discover_files, ingest_all
from src.rag.regulation import search_by_term, search_regulations
from src.rag.reranker import dedup_near_duplicates, rerank
from src.rag.retriever import (
    count_documents,
    find_unindexed_sources,
    index_documents,
    keyword_search,
    reset_collection,
    search_with_score,
)

__all__ = [
    "ingestion",
    "parent_store",
    "retriever",
    "discover_files",
    "ingest_all",
    "search_by_term",
    "search_regulations",
    "dedup_near_duplicates",
    "rerank",
    "count_documents",
    "find_unindexed_sources",
    "index_documents",
    "keyword_search",
    "reset_collection",
    "search_with_score",
]
