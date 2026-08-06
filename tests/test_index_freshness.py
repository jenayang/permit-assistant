"""인덱스 최신성 점검 회귀 테스트.

벡터DB를 실제로 띄우지 않고, discover_files/indexed_sources만 대체해서
비교 로직 자체를 검증한다(느린 임베딩 로드 없이 결정론적으로 실행).

배경(2026-08-02): data/에 법령 6종을 추가하고 재인덱싱을 잊은 채 운영되고
있었다. 검색은 실패하지 않고 "가장 비슷한" 무관한 조문을 돌려주기 때문에
답변은 그럴듯한데 근거만 틀린 상태였고, 그래서 아무도 눈치채지 못했다.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.rag import retriever


@pytest.fixture
def fake_data(monkeypatch):
    """discover_files/indexed_sources를 주입 가능하게 만든다."""
    def setup(files: list[str], indexed: set[str]):
        import src.rag.ingestion as ingestion
        monkeypatch.setattr(
            ingestion, "discover_files",
            lambda *a, **k: [config.DATA_DIR / f for f in files],
        )
        monkeypatch.setattr(retriever, "indexed_sources", lambda: indexed)
    return setup


def test_reports_files_missing_from_index(fake_data):
    """data/엔 있는데 DB에 없는 파일을 잡아내야 한다 - 이게 이 점검의 존재 이유."""
    fake_data(
        files=["laws/건축법.pdf", "laws/부가가치세법.pdf"],
        indexed={"laws/건축법.pdf"},
    )
    assert retriever.find_unindexed_sources() == ["laws/부가가치세법.pdf"]


def test_no_warning_when_index_is_current(fake_data):
    """전부 인덱싱돼 있으면 빈 목록이어야 한다 - 정상 상태에서 매번 경고가
    뜨면 경고 자체가 무시된다."""
    fake_data(
        files=["laws/건축법.pdf", "laws/부가가치세법.pdf"],
        indexed={"laws/건축법.pdf", "laws/부가가치세법.pdf"},
    )
    assert retriever.find_unindexed_sources() == []


def test_extra_indexed_sources_are_not_reported(fake_data):
    """DB에만 있고 data/엔 없는 항목(삭제된 문서)은 이 점검 대상이 아니다 -
    증분 인덱싱이 알아서 정리하며, 여기서 같이 보고하면 신호가 흐려진다."""
    fake_data(files=["laws/건축법.pdf"], indexed={"laws/건축법.pdf", "laws/옛날법.pdf"})
    assert retriever.find_unindexed_sources() == []


def test_source_path_is_relative_to_data_dir(fake_data):
    """비교 키는 인제스천이 metadata["source"]에 넣는 값과 같은 형식
    (data/ 기준 상대경로)이어야 한다 - 형식이 어긋나면 전부 미인덱싱으로
    잘못 잡혀 경고가 무의미해진다."""
    fake_data(files=["ordinances/서울시 건축조례.pdf"], indexed=set())
    assert retriever.find_unindexed_sources() == ["ordinances/서울시 건축조례.pdf"]
    assert not Path(retriever.find_unindexed_sources()[0]).is_absolute()
