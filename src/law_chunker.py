"""법령/조례 특화 청킹.

일반 텍스트와 달리 법령은 조항 단위로 자르는 게 검색에 유리.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# 조항 패턴 (제XX조, 제XX조의YY 등 지원)
# 예: "제56조", "제 56 조", "제56조의2", "제56조(제목)"
ARTICLE_PATTERN = re.compile(
    r'(?:^|\n)\s*제\s*(\d+)\s*조(?:의\s*(\d+))?(?:\s*\(([^)]+)\))?',
    re.MULTILINE,
)

# 최대 청크 크기 (너무 긴 조항은 예외 처리 필요)
MAX_ARTICLE_SIZE = 3000

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
        
        # 너무 긴 조항은 크기 제한 (완벽하진 않지만 실용적)
        if len(content) > MAX_ARTICLE_SIZE:
            content = content[:MAX_ARTICLE_SIZE] + "...[이하 생략]"
            logger.warning(
                "긴 조항 잘림: 제%d조 (%d자 → %d자)",
                article_num, len(content), MAX_ARTICLE_SIZE,
            )
        
        articles.append({
            "article_num": article_num,
            "article_sub": article_sub,
            "title": title,
            "content": content,
        })
    
    return articles


def split_law_document(doc: Document) -> list[Document]:
    """법령 Document를 조항 단위로 분할."""
    articles = extract_articles(doc.page_content)
    
    chunks = []
    for article in articles:
        # 조항 번호 문자열 만들기 (예: "56", "56-2")
        article_id = str(article["article_num"])
        if article["article_sub"] is not None:
            article_id += f"-{article['article_sub']}"
        
        chunk = Document(
            page_content=article["content"],
            metadata={
                **doc.metadata,   # 기존 metadata 유지
                "article_num": article["article_num"],
                "article_sub": article["article_sub"],
                "article_id": article_id,
                "article_title": article["title"],
            },
        )
        chunks.append(chunk)
    
    return chunks


def split_law_documents(docs: list[Document]) -> list[Document]:
    """여러 Document를 조항 단위로 분할 (페이지가 여러 개인 경우)."""
    # 페이지들을 하나로 합쳐서 조항 분리
    # (조항이 페이지 경계를 넘길 수 있으므로)
    if not docs:
        return []
    
    # 모든 페이지 합치기
    full_text = "\n".join(d.page_content for d in docs)
    base_metadata = docs[0].metadata.copy()
    
    combined_doc = Document(
        page_content=full_text,
        metadata=base_metadata,
    )
    
    return split_law_document(combined_doc)