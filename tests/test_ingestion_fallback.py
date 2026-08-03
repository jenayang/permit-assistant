"""조문 구조가 없는 법령 문서의 일반 청킹 폴백 회귀 테스트.

배경(2026-08-02): 별지 서식류는 `is_law_document()`가 참이라 법령 청킹기로
가는데, 조문(제N조) 구조가 없어 0청크가 나오고 **본문이 멀쩡히 추출됐는데도
통째로 버려졌다.** 검색에서 영영 사라지므로 "교육위탁 신청서 처리기간이
얼마인가요?", "실명제 마크 규격이 어떻게 되나요?" 같은 질문에 답할 수 없었다.
조문 단위 부모-자식 구조는 못 만들지만, 아예 못 찾는 것보다는 낫다.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from src.rag import ingestion


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={})


def test_law_document_without_articles_falls_back(monkeypatch):
    """법령으로 분류됐지만 조문이 없으면 일반 청킹으로 되살려야 한다."""
    body = "[별지 제1호 서식] 교육위탁 지정 신청서 처리기간 30일 신청자 성명 생년월일 업소명"
    monkeypatch.setattr(ingestion, "load_document", lambda p: [_doc(body)])
    monkeypatch.setattr(ingestion, "preprocess_documents", lambda d: d)
    monkeypatch.setattr(ingestion, "is_law_document", lambda p: True)
    monkeypatch.setattr(ingestion, "split_law_documents", lambda *a, **k: ([], []))

    chunks, parents = ingestion.process_file(Path("ordinances/별지1.hwp"))

    assert chunks, "조문이 없다는 이유로 문서가 통째로 버려지면 안 된다"
    assert "처리기간 30일" in "".join(c.page_content for c in chunks)
    assert parents == []  # 조문 단위 부모-자식 구조는 만들 수 없다


def test_law_document_with_articles_keeps_law_chunking(monkeypatch):
    """조문이 정상적으로 잡히면 폴백이 끼어들면 안 된다 - 부모-자식 구조를
    잃으면 조항 전체를 보여주는 검색 품질이 떨어진다."""
    law_chunk = _doc("제20조(부설주차장의 설치기준) ...")
    parents = [{"source": "x", "article_id": "20", "content": "제20조 전문"}]
    monkeypatch.setattr(ingestion, "load_document", lambda p: [_doc("제20조 ...")])
    monkeypatch.setattr(ingestion, "preprocess_documents", lambda d: d)
    monkeypatch.setattr(ingestion, "is_law_document", lambda p: True)
    monkeypatch.setattr(ingestion, "split_law_documents", lambda *a, **k: ([law_chunk], parents))

    chunks, got_parents = ingestion.process_file(Path("ordinances/조례.pdf"))

    assert len(chunks) == 1
    assert got_parents == parents


def test_non_law_document_uses_general_split(monkeypatch):
    """법령이 아닌 문서는 원래대로 일반 청킹을 쓴다(동작 변화 없음)."""
    monkeypatch.setattr(ingestion, "load_document", lambda p: [_doc("일반 안내 문서 본문")])
    monkeypatch.setattr(ingestion, "preprocess_documents", lambda d: d)
    monkeypatch.setattr(ingestion, "is_law_document", lambda p: False)

    chunks, parents = ingestion.process_file(Path("procedures/안내서.pdf"))

    assert chunks
    assert parents == []
