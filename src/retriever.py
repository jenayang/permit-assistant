"""임베딩 + 벡터DB + 검색.

LangChain Chroma + HuggingFace 임베딩 사용.
임베딩 모델: paraphrase-multilingual-MiniLM-L12-v2 (다국어 지원)
"""
# v1 : sentence-transformers로 임베딩 → ChromaDB(로컬 영구 저장) → 코사인 검색.

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.vectorstores import VectorStoreRetriever 
from src import config

logger = logging.getLogger(__name__)


# === 임베딩 모델 (모듈 로드 시 1회 생성) ===(1. 임베딩 모델 로드)
embeddings = HuggingFaceEmbeddings(
        model_name = config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# === Vectorstore 초기화 ===(2. Chroma 인스턴스 반환)
_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    """ChromaDB 벡터스토어 객체 반환 (모듈 레벨 캐싱).

    persist_directory 지정 → 디스크에 자동 저장/로드.
    매 호출마다 새 Chroma()를 생성하면 LangGraph가 도구를 병렬 실행할 때
    chromadb의 SharedSystemClient 내부 딕셔너리에 경쟁 상태(race condition)가
    생겨 KeyError가 발생할 수 있어 싱글턴으로 재사용한다.
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name = config.COLLECTION_NAME,
            embedding_function = embeddings, # HuggingFace 임베딩 모델 사용
            persist_directory=str(config.CHROMA_DIR), # 디스크 자동저장
            collection_metadata={"hnsw:space": config.DISTANCE_METRIC}, # 코사인 유사도
        )
    return _vectorstore


# === 청크 추가 (인덱싱) === (3-1. 인덱싱)
def add_documents(docs: list[Document]) -> None :
    """Document 청크들을 벡터스토어에 추가. """
    if not docs:
        logger.warning("추가할 문서가 없습니다.")
        return

    vectorstore = get_vectorstore() # Chroma 임베딩 객체생성

    # ID 생성 (source + chunk_index로 고유성 보장)
    ids = [
        f"{d.metadata['source']}_{d.metadata['chunk_index']}"
        for d in docs
    ]
    vectorstore.add_documents(documents=docs, ids=ids)  # docs에 ids 넣어서 임베딩 DB 저장(실행)
    logger.info("벡터스토어에 %d개 문서 추가됨", len(docs))


# === 검색 ===(3-2. 검색, 검색+점수) # 현재는 retriever써서 사용안함. but 디버깅/평가용으로 사용을 위함
def search(query: str, k: int = config.TOP_K) -> list[Document]:
    """쿼리와 유사한 청크 top-k 검색."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k=k)  # 쿼리 임베딩 + 유사도 검사

def search_with_score(query: str, k: int = config.TOP_K) -> list[tuple[Document, float]]:
    """검색 + 유사도 점수 함께 반환."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(query, k=k)


# === 키워드(정확 매칭) 검색 ===
# 법률 용어의 정의를 물을 때는 유사도 검색보다 유리함.
# (조문은 딱딱한 문어체라 자연어 질문과의 임베딩 유사도가 낮게 나오는 경우가 많음)
def keyword_search(keyword: str, k: int = config.TOP_K) -> list[Document]:
    """키워드를 포함하는 청크를 부분 문자열 매칭으로 검색.

    조항 제목에 키워드가 포함된 조항(정의 조항일 가능성 높음)을 우선 정렬.
    """
    vectorstore = get_vectorstore()
    result = vectorstore.get(
        where_document={"$contains": keyword},
        include=["documents", "metadatas"],
    )

    docs = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(result["documents"], result["metadatas"])
    ]

    docs.sort(key=lambda d: keyword not in (d.metadata.get("article_title") or ""))

    return docs[:k]



# === Retriever 객체 (LCEL 체인용) ===(3-3. LCEL용)
# 체인에 들어갈 retriever !!!! -> generator로 연결
def get_retriever(k: int = config.TOP_K) -> VectorStoreRetriever: # 검색기 객체 생성만 -> 실행은 나중에 get.retriever().invoke(query)할때 실행됨
    """LCEL 체인에 끼울 수 있는 Retriever 반환."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k}) # Retriever 생성(as_retriever)



# === 컬렉션 관리 ===(관리용)
def count_documents() -> int:
    """현재 컬렉션의 문서 개수."""
    vectorstore = get_vectorstore()
    return vectorstore._collection.count() # 카운트

def reset_collection() -> None:
    """컬렉션 전체 삭제 후 재생성. (재인덱싱 시 사용)"""
    vectorstore = get_vectorstore()
    try:
        vectorstore.delete_collection() # 삭제
        logger.info("기존 컬렉션 삭제 완료")
    except Exception as e:
        logger.warning("컬렉션 삭세 실패 (없을 수도 있음): %s", e)

