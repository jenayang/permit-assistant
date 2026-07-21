"""부모-자식 청킹의 부모(조항 전체) 저장소.

임베딩/검색은 토큰 한도에 맞춘 작은 자식 청크로 하되(law_chunker 참고),
LLM에 넘길 때는 검색된 자식이 속한 조항 전체(잘리지 않은 원문)를 보여주기
위한 조회용 SQLite 저장소.

article_id 기준으로 우리 법령 데이터 구조에 맞춰 직접 구현한 버전(하드코딩).
나중에 문서 타입이 다양해지면 LangChain의 ParentDocumentRetriever로
리팩토링 예정.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

_DB_PATH = config.CHROMA_DIR / "parent_articles.sqlite"


def _get_conn() -> sqlite3.Connection:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parent_articles (
            source TEXT NOT NULL,
            article_id TEXT NOT NULL,
            article_title TEXT,
            content TEXT NOT NULL,
            PRIMARY KEY (source, article_id)
        )
        """
    )
    return conn


def replace_all(records: list[dict]) -> None:
    """부모 조항 테이블을 통째로 새로 씀 (매 ingest마다 호출).

    조항 수가 수백 개 수준으로 적어서, 증분 갱신 대신 매번 전체를
    지우고 다시 쓰는 게 더 단순하고 충분히 빠르다.

    INSERT OR REPLACE 사용: (source, article_id)가 드물게 중복되는 경우
    (예: "OO조부터 XX조까지" 같은 인용이 줄바꿈 때문에 가짜 조항 헤더로
    오인식되는 극히 일부 케이스) PRIMARY KEY 충돌로 인덱싱 전체가
    크래시나지 않도록 방어한다. 마지막 값으로 덮어써진다.
    """
    conn = _get_conn()
    with conn:
        conn.execute("DELETE FROM parent_articles")
        conn.executemany(
            "INSERT OR REPLACE INTO parent_articles (source, article_id, article_title, content) "
            "VALUES (:source, :article_id, :article_title, :content)",
            records,
        )
    conn.close()
    logger.info("부모 조항 레코드 %d개 저장 완료", len(records))


def get_parent(source: str, article_id: str) -> str | None:
    """(source, article_id)에 해당하는 조항 전체 원문 반환."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT content FROM parent_articles WHERE source = ? AND article_id = ?",
        (source, article_id),
    ).fetchone()
    conn.close()
    return row[0] if row else None
