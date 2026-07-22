"""임베딩 + 벡터DB + 검색.

LangChain Chroma + HuggingFace 임베딩 사용.
임베딩 모델: intfloat/multilingual-e5-base (다국어 지원, 512토큰)
"""
# v1 : sentence-transformers로 임베딩 → ChromaDB(로컬 영구 저장) → 코사인 검색.

from __future__ import annotations

import logging
import shutil
import sqlite3
import threading

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_classic.indexes import SQLRecordManager, index as lc_index
from src import config

logger = logging.getLogger(__name__)


# === 임베딩 모델 (모듈 로드 시 1회 생성) ===(1. 임베딩 모델 로드)
# e5 계열 모델은 "query: "/"passage: " 접두사를 붙여야 학습 때와 같은 조건으로
# 동작함 (안 붙이면 에러는 안 나지만 검색 품질이 떨어짐). embed_documents()는
# encode_kwargs를, embed_query()는 query_encode_kwargs를 쓰므로 문서/질문에
# 서로 다른 접두사를 자동으로 붙일 수 있다.
embeddings = HuggingFaceEmbeddings(
        model_name = config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "},
    )


# === 토크나이저 조회 (law_chunker의 토큰 기준 서브 청킹용) ===
def get_tokenizer():
    """임베딩 모델의 토크나이저 반환."""
    return embeddings._client.tokenizer


def get_max_chunk_tokens() -> int:
    """청크 하나의 최대 토큰 수 (임베딩 모델 한도와 실용적 상한 중 작은 값).

    이 값을 기준으로 청킹하면, 모델을 바꿔도(예: 128 → 512토큰) 코드 수정 없이
    자동으로 새 모델의 한계에 맞춰 청크 크기가 조정된다. 다만 모델 한도가
    아주 크더라도(예: 8192토큰), 검색 정밀도와 LLM 컨텍스트 예산을 위해
    config.MAX_PRACTICAL_CHUNK_TOKENS를 넘지 않도록 상한을 둔다.
    """
    return min(embeddings._client.max_seq_length, config.MAX_PRACTICAL_CHUNK_TOKENS)


# === Vectorstore 초기화 ===(2. Chroma 인스턴스 반환)
_vectorstore: Chroma | None = None
_vectorstore_lock = threading.Lock()


def get_vectorstore() -> Chroma:
    """ChromaDB 벡터스토어 객체 반환 (모듈 레벨 캐싱).

    persist_directory 지정 → 디스크에 자동 저장/로드.
    매 호출마다 새 Chroma()를 생성하면 LangGraph가 도구를 병렬 실행할 때
    chromadb의 SharedSystemClient 내부 딕셔너리에 경쟁 상태(race condition)가
    생겨 KeyError가 발생할 수 있어 싱글턴으로 재사용한다.

    싱글턴 초기화 자체도 락으로 감싼다: FastAPI는 요청마다 별도 스레드에서
    처리하므로, 락 없이 "None이면 생성"만 하면 첫 호출이 겹칠 때(예: 여러
    요청이 동시에 처음 들어옴) 두 스레드가 동시에 Chroma()를 만들려다가
    위와 같은 SharedSystemClient 경쟁 상태가 그대로 재현된다(실제로 겪음:
    'RustBindingsAPI' object has no attribute 'bindings' 에러).
    """
    global _vectorstore
    if _vectorstore is None:
        with _vectorstore_lock:
            if _vectorstore is None:  # 락 대기 중 다른 스레드가 이미 만들었을 수 있음
                _vectorstore = Chroma(
                    collection_name = config.COLLECTION_NAME,
                    embedding_function = embeddings, # HuggingFace 임베딩 모델 사용
                    persist_directory=str(config.CHROMA_DIR), # 디스크 자동저장
                    collection_metadata={"hnsw:space": config.DISTANCE_METRIC}, # 코사인 유사도
                )
    return _vectorstore


# === 레코드 매니저 (증분 인덱싱용 해시 기록) ===
_record_manager: SQLRecordManager | None = None
_record_manager_lock = threading.Lock()


def get_record_manager() -> SQLRecordManager:
    """청크별 콘텐츠 해시를 SQLite에 기록하는 레코드 매니저 (모듈 레벨 싱글턴).

    index_documents()가 이걸로 "지난번과 내용이 같은 청크"는 재임베딩을
    건너뛰고, "이번엔 안 나온(삭제/축소된) 청크"는 벡터스토어에서 자동으로 지운다.
    get_vectorstore()와 같은 이유로 초기화를 락으로 감싼다.
    """
    global _record_manager
    if _record_manager is None:
        with _record_manager_lock:
            if _record_manager is None:
                _record_manager = SQLRecordManager(
                    namespace=f"chroma/{config.COLLECTION_NAME}",
                    db_url=f"sqlite:///{config.CHROMA_DIR / 'record_manager.sqlite'}",
                )
                _record_manager.create_schema()
    return _record_manager


# === 청크 추가 (증분 인덱싱) === (3-1. 인덱싱)
def index_documents(docs: list[Document]) -> dict:
    """Document 청크들을 해시 기반으로 증분 인덱싱.

    내용이 바뀌지 않은 청크는 재임베딩을 스킵하고, source 기준으로
    이번 호출에 포함된 파일들 중 없어진 청크는 자동 삭제한다.

    cleanup="scoped_full" 사용: "incremental"은 100개 단위 배치 처리 도중
    바로바로 청소를 실행해서, 한 파일의 청크 수가 배치 크기(100)보다 많으면
    아직 처리 안 된 뒷부분 청크를 "없어진 청크"로 오인해 삭제 후 재추가하는
    버그가 있음(batch_size에 의존하는 숨은 가정이 생김). scoped_full은 청소를
    전체 배치 처리가 끝난 뒤, 이번 호출에 실제로 포함된 source로만 한정해서
    실행하므로 이 문제가 없고, 나중에 파일 단위 부분 인덱싱을 추가해도 안전하다
    (미포함 source는 절대 건드리지 않음 — cleanup="full"과의 차이).
    """
    if not docs:
        logger.warning("추가할 문서가 없습니다.")
        return {"num_added": 0, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}

    vectorstore = get_vectorstore()
    record_manager = get_record_manager()

    result = lc_index(
        docs,
        record_manager,
        vectorstore,
        cleanup="scoped_full",
        source_id_key="source",
    )
    logger.info(
        "인덱싱 결과: 추가 %d, 갱신 %d, 스킵(변경없음) %d, 삭제 %d",
        result["num_added"], result["num_updated"], result["num_skipped"], result["num_deleted"],
    )
    return result


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

def _cleanup_orphaned_segments() -> None:
    """chromadb가 못 지운 예전 세그먼트 디렉토리를 카탈로그 기준으로 직접 정리.

    chromadb는 세그먼트가 "현재 프로세스 메모리에 이미 로드돼 있을 때만"
    디스크 파일을 지운다. 우리는 --ingest를 매번 새 CLI 프로세스로 돌리다
    보니 예전 리셋 때 만들어진 세그먼트는 이 프로세스 메모리에 없어서
    delete_collection()이 카탈로그(sqlite)에서만 지우고 디스크 폴더는
    고아로 남긴다. chroma.sqlite3의 segments 테이블에 없는 UUID 폴더를
    직접 지워서 이걸 보완한다.
    """
    db_path = config.CHROMA_DIR / "chroma.sqlite3"
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    valid_ids = {row[0] for row in conn.execute("SELECT id FROM segments")}
    conn.close()

    for entry in config.CHROMA_DIR.iterdir():
        if entry.is_dir() and entry.name not in valid_ids:
            shutil.rmtree(entry)
            logger.info("고아 세그먼트 폴더 삭제: %s", entry.name)


def reset_collection() -> None:
    """컬렉션 전체 삭제 후 재생성 + 레코드 매니저 해시 기록도 초기화. (재인덱싱 시 사용)

    해시 기록을 같이 지우지 않으면, 벡터스토어는 비었는데 레코드 매니저는
    "이미 인덱싱됨"으로 착각해서 index_documents()가 재임베딩을 스킵해버린다.
    """
    vectorstore = get_vectorstore()
    try:
        vectorstore.reset_collection() # 삭제 + 빈 컬렉션으로 재생성 (delete_collection만 쓰면 재생성이 안 돼서 이후 add 시 에러남)
        logger.info("기존 컬렉션 삭제 완료")
    except Exception as e:
        logger.warning("컬렉션 삭제 실패 (없을 수도 있음): %s", e)

    _cleanup_orphaned_segments()

    record_manager = get_record_manager()
    keys = record_manager.list_keys()
    if keys:
        record_manager.delete_keys(keys)
        logger.info("레코드 매니저 해시 기록 %d개 삭제 완료", len(keys))

