"""법령/조례 특화 청킹.

일반 텍스트와 달리 법령은 조항 단위로 자르는 게 검색에 유리.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# 조항 패턴 (제XX조, 제XX조의YY 등 지원)
# 예: "제56조", "제 56 조", "제56조의2", "제56조(제목)"
ARTICLE_PATTERN = re.compile(
    r'(?:^|\n)\s*제\s*(\d+)\s*조(?:의\s*(\d+))?(?:\s*\(([^)]+)\))?',
    re.MULTILINE,
)

# 항 마커 (①②③...⑳)
PARAGRAPH_MARKERS = [chr(0x2460 + i) for i in range(20)]

# 호 마커 (1. 2. 3...)
NUMBERED_ITEM_MARKERS = [f"{i}." for i in range(1, 31)]

# 목 마커 (가. 나. 다...) - 법령 관용 순서(14개, 일부 자음 생략)
LETTER_SEQUENCE = "가나다라마바사아자차카타파하"
LETTERED_ITEM_MARKERS = [f"{ch}." for ch in LETTER_SEQUENCE]

# 삭제/생략된 조항 (예: "제35조 삭제 <2019. 4. 30.>", "제8조 생략")
# 실질 내용이 없어 임베딩 노이즈만 유발 → 인덱싱에서 제외
REPEALED_ARTICLE_PATTERN = re.compile(
    r'^제\s*\d+\s*조(?:의\s*\d+)?'
    r'(?:\s*(?:부터|내지)\s*제\s*\d+\s*조(?:의\s*\d+)?\s*까지)?'
    r'\s*(?:삭제|생략)\s*(?:<[^>]*>)?\s*$'
)


def is_repealed_article(content: str) -> bool:
    """실질 내용 없이 삭제/생략만 표시된 조항인지 판단."""
    return bool(REPEALED_ARTICLE_PATTERN.match(content.strip()))


def is_law_document(filepath: Path) -> bool:
    """법령/조례 문서인지 판단."""
    # data/laws/ 또는 data/ordinances/ 폴더의 파일
    parts = filepath.parts
    return any(cat in parts for cat in ("laws", "ordinances"))


def extract_articles(text: str) -> list[dict]:
    """텍스트에서 조항 단위로 분리.
    
    Returns:
        [{"article_num": 56, "article_sub": None, "title": "건축물의 용적률", "content": "..."}]
    """
    matches = list(ARTICLE_PATTERN.finditer(text))
    
    if not matches:
        # 조항 패턴이 없으면 전체를 하나로
        return [{
            "article_num": 0,
            "article_sub": None,
            "title": "",
            "content": text.strip(),
        }]
    
    articles = []
    
    # 첫 조항 앞의 내용 (목차, 개정 이력 등)은 별도 청크로 (스킵할 수도 있음)
    if matches[0].start() > 100:
        preamble = text[:matches[0].start()].strip()
        if len(preamble) > 200:   # 의미 있는 크기만
            articles.append({
                "article_num": 0,
                "article_sub": None,
                "title": "머리말",
                "content": preamble,
            })
    
    # 조항별로 분리
    for i, match in enumerate(matches):
        article_num = int(match.group(1))
        article_sub = int(match.group(2)) if match.group(2) else None
        title = (match.group(3) or "").strip()
        
        # 다음 조항까지가 이 조항의 내용
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        articles.append({
            "article_num": article_num,
            "article_sub": article_sub,
            "title": title,
            "content": content,
        })
    
    return articles


def _split_for_embedding(text: str, tokenizer, max_tokens: int, overlap: int) -> list[str]:
    """조항 텍스트가 max_tokens를 넘으면 임베딩 가능한 크기로 서브 분할.

    법령 구조 순서(문단/줄바꿈 → 항 → 호 → 목)를 우선 시도하고, 그래도 안
    맞으면 문장/공백 단위로, 최후엔 글자 수로 강제 분할한다.
    RecursiveCharacterTextSplitter는 한 구분자로 나눈 조각이 여전히 크면
    그 조각만 다음 구분자로 재귀적으로 다시 나누므로, 어떤 입력이든 결국
    max_tokens 이내로 들어온다.
    """
    if len(tokenizer.encode(text)) <= max_tokens:
        return [text]

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        separators=[
            "\n\n", "\n",
            *PARAGRAPH_MARKERS,       # 항: ①②③...
            *NUMBERED_ITEM_MARKERS,   # 호: 1. 2. 3...
            *LETTERED_ITEM_MARKERS,   # 목: 가. 나. 다...
            "다. ", " ", "",
        ],
    )
    return splitter.split_text(text)


def split_law_document(
    doc: Document,
    tokenizer=None,
    max_tokens: int | None = None,
    overlap: int = 0,
) -> tuple[list[Document], list[dict]]:
    """법령 Document를 조항 단위로 분할.

    Returns:
        (자식 청크 리스트, 부모 조항 레코드 리스트).
        - 자식 청크: 임베딩/검색용. tokenizer + max_tokens가 주어지면 조항이
          토큰 한도를 넘을 때 서브 분할되고, 같은 article_id를 공유한다.
        - 부모 레코드: 조항 전체 원문(잘리지 않음). article_id로 자식 청크와
          연결되며, parent_store에 저장해서 검색 시 전체 맥락 복원에 쓴다.
    """
    articles = extract_articles(doc.page_content)
    source = doc.metadata.get("source")

    child_chunks: list[Document] = []
    parent_records: list[dict] = []
    seen_article_ids: dict[str, int] = {}

    for article in articles:
        # 조항 번호 문자열 만들기 (예: "56", "56-2")
        base_id = str(article["article_num"])
        if article["article_sub"] is not None:
            base_id += f"-{article['article_sub']}"

        # 부칙이 본칙과 별개로 "제1조"부터 번호를 다시 매기는 경우가 흔해서,
        # 같은 문서 안에서 같은 번호가 재등장하면 구분자를 붙여 고유화한다.
        occurrence = seen_article_ids.get(base_id, 0)
        seen_article_ids[base_id] = occurrence + 1
        article_id = base_id if occurrence == 0 else f"{base_id}-dup{occurrence}"

        parent_records.append({
            "source": source,
            "article_id": article_id,
            "article_title": article["title"],
            "content": article["content"],
        })

        base_metadata = {
            **doc.metadata,   # 기존 metadata 유지
            "article_num": article["article_num"],
            "article_sub": article["article_sub"],
            "article_id": article_id,
            "article_title": article["title"],
        }

        if tokenizer is not None and max_tokens is not None:
            pieces = _split_for_embedding(article["content"], tokenizer, max_tokens, overlap)
        else:
            pieces = [article["content"]]

        for sub_idx, piece in enumerate(pieces):
            child_chunks.append(Document(
                page_content=piece,
                metadata={**base_metadata, "sub_chunk_index": sub_idx},
            ))

    return child_chunks, parent_records


def split_law_documents(
    docs: list[Document],
    tokenizer=None,
    max_tokens: int | None = None,
    overlap: int = 0,
) -> tuple[list[Document], list[dict]]:
    """여러 Document를 조항 단위로 분할 (페이지가 여러 개인 경우)."""
    # 페이지들을 하나로 합쳐서 조항 분리
    # (조항이 페이지 경계를 넘길 수 있으므로)
    if not docs:
        return [], []

    # 모든 페이지 합치기
    full_text = "\n".join(d.page_content for d in docs)
    base_metadata = docs[0].metadata.copy()

    combined_doc = Document(
        page_content=full_text,
        metadata=base_metadata,
    )

    return split_law_document(combined_doc, tokenizer=tokenizer, max_tokens=max_tokens, overlap=overlap)