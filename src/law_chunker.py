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


# 조항 패턴 (제XX조, 제XX조의YY 등 지원). 제목 괄호를 필수로 요구해서
# "법 제52조부터 제57조까지" 같은 인용 참조가 줄바꿈 때문에 "제57조"부터
# 새 줄로 시작해도 가짜 조항 헤더로 안 잡히게 한다(제목 없이 인용만 있음).
# 부작용: "제42조 삭제 <1999. 4. 30.>"처럼 제목이 원래 없는 삭제/생략
# 조항도 안 잡히는데, 이건 의도된 것 - 실질 내용이 없어 어차피 인덱싱할
# 필요가 없고, 앞 조항 내용에 조용히 흡수된다.
# 예: "제56조(용적률)", "제56조의2(대지면적)"
ARTICLE_PATTERN = re.compile(
    r'(?:^|\n)\s*제\s*(\d+)\s*조(?:의\s*(\d+))?\s*\(([^)]+)\)',
    re.MULTILINE,
)

# 항 마커 (①②③...⑳) -- PyMuPDFLoader 인식 오류 -- (사실상 무효.. 일단 살려둠..)
PARAGRAPH_MARKERS = [chr(0x2460 + i) for i in range(20)]

# 호 마커 (1. 2. 3...)
NUMBERED_ITEM_MARKERS = [f"{i}." for i in range(1, 31)]

# 목 마커 (가. 나. 다...) - 법령 관용 순서(14개, 일부 자음 생략)
LETTER_SEQUENCE = "가나다라마바사아자차카타파하"
LETTERED_ITEM_MARKERS = [f"{ch}." for ch in LETTER_SEQUENCE]

# 별표/별지서식 헤더 (본문 중 "별표 2와 같다" 같은 참조는 대괄호가 없어서 안 걸리고,
# 실제 첨부 표/서식이 시작되는 자리만 "[별표 2]"처럼 대괄호로 나온다 - 파일명에도, 본문에도 공통)
ATTACHMENT_LABEL_PATTERN = re.compile(r'\[별표\s*(\d+)\]|\[별지\s*제?\s*(\d+)\s*호\s*서식\]')
ATTACHMENT_ARTICLE_REF_PATTERN = re.compile(r'제\s*(\d+)\s*조')

# 진짜 부칙 표제 (예: "부칙 <제35811호,2025. 10. 1.>"). 그냥 "부칙"으로 찾으면
# 본문 중 "OO법 일부개정법률 부칙 제9조에 따라" 같은 인용까지 걸려서, 부칙
# 표제에만 붙는 "<공포번호, 날짜>" 꺾쇠괄호까지 포함해서 구분한다.
BUCHIK_PATTERN = re.compile(r'부\s*칙\s*<')


def _attachment_article_id(filename: str) -> str | None:
    """파일명만으로 된 별표/별지서식 문서의 article_id 생성.

    extract_articles()가 본문에서 "제XX조" 조항 헤더를 못 찾으면(별표 파일은
    본문이 표 제목뿐이라 조항 형식이 아님) article_num=0으로 폴백하는데,
    그대로 두면 인용 시 "[출처: ... 제0조]"처럼 틀린 조항 번호가 나온다.
    대신 파일명의 "[별표2]"와 "(제20조 관련)" 표기를 파싱해 "20-별표2" 같은
    식별자를 만든다. 조항 참조가 없거나(예: 오타로 "조"가 빠진 경우) 별표
    번호만 있으면 "별표2"만 반환.
    """
    label_match = ATTACHMENT_LABEL_PATTERN.search(filename)
    if not label_match:
        return None
    label = f"별표{label_match.group(1)}" if label_match.group(1) else f"별지{label_match.group(2)}호"

    article_match = ATTACHMENT_ARTICLE_REF_PATTERN.search(filename)
    if article_match:
        return f"{article_match.group(1)}-{label}"
    return label


def is_law_document(filepath: Path) -> bool:
    """법령/조례 문서인지 판단."""
    # data/laws/ 또는 data/ordinances/ 폴더의 파일
    parts = filepath.parts
    return any(cat in parts for cat in ("laws", "ordinances"))


def _extract_attachments(text: str) -> list[dict]:
    """별표 구역을 대괄호 헤더(예: "[별표 2]") 기준으로 분리. 별지서식은 제외.

    이 구역엔 "제113조제2항제1호" 같은 조항 참조가 표 데이터로 많이 섞여
    있어서 ARTICLE_PATTERN으로 자르면 표 행을 가짜 조항으로 오인식한다.
    그래서 조항 번호가 아니라 별표 번호로 구분한다.

    별지서식(제출용 신청서 양식)은 경계 계산에는 쓰되(그래야 별표 내용에
    안 섞임) 실제로 인덱싱하지는 않는다 - 규정 내용이 아니라 빈 양식이라
    챗봇 답변에 필요 없음.
    """
    matches = list(ATTACHMENT_LABEL_PATTERN.finditer(text))
    attachments = []
    for i, match in enumerate(matches):
        if match.group(1) is None:
            continue  # 별지서식은 스킵
        label = f"별표{match.group(1)}"
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        attachments.append({
            "article_id": label,
            "title": label,
            "content": text[start:end].strip(),
        })
    return attachments


def _extract_buchik(text: str) -> list[dict]:
    """부칙 구역을 분리.

    부칙 자체가 "제1조(시행일)"부터 번호를 다시 매기는 경우가 흔해서, 본문의
    진짜 제1조와 article_id가 겹치지 않도록 "부칙-" 접두사로 네임스페이스를
    분리한다. 부칙 안에 조항 구조가 없으면(예: "이 조례는 공포한 날부터
    시행한다" 한 줄짜리) 통째로 "부칙" 하나로 묶는다.
    """
    matches = list(ARTICLE_PATTERN.finditer(text))
    if not matches:
        return [{"article_id": "부칙", "title": "부칙", "content": text.strip()}] if text.strip() else []

    buchik = []
    for i, match in enumerate(matches):
        num = int(match.group(1))
        sub = match.group(2)
        label = f"부칙-{num}" + (f"-{sub}" if sub else "")
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        buchik.append({
            "article_id": label,
            "title": (match.group(3) or "").strip(),
            "content": text[start:end].strip(),
        })
    return buchik


def extract_articles(text: str) -> list[dict]:
    """텍스트에서 본문(조항) / 부칙 / 별표·별지서식 구역으로 나눠 분리.

    - 본문: ARTICLE_PATTERN("제XX조")으로 조항별 분리
    - 부칙: _extract_buchik()으로 별도 네임스페이스("부칙-1" 등) 처리
      (부칙이 자체적으로 "제1조"부터 다시 매겨서, 본문과 같은 정규식 범위에
      두면 진짜 제1조와 article_id가 충돌한다)
    - 별표·별지서식: _extract_attachments()로 별도 처리 (표 안의 조항 참조를
      가짜 조항으로 오인식하는 걸 방지)

    Returns:
        [{"article_num": 56, "article_sub": None, "title": "건축물의 용적률", "content": "..."}]
        (부칙/별표/서식 항목은 article_num 대신 "article_id"를 직접 갖고 있음)
    """
    attachment_match = ATTACHMENT_LABEL_PATTERN.search(text)
    attachment_start = attachment_match.start() if attachment_match else len(text)

    buchik_match = BUCHIK_PATTERN.search(text[:attachment_start])
    body_end = buchik_match.start() if buchik_match else attachment_start

    matches = list(ARTICLE_PATTERN.finditer(text[:body_end]))

    articles = []

    if not matches:
        # 조항 패턴이 없으면 본문 전체를 하나로
        body_text = text[:body_end].strip()
        if body_text:
            articles.append({
                "article_num": 0,
                "article_sub": None,
                "title": "",
                "content": body_text,
            })
    else:
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

            # 다음 조항까지가 이 조항의 내용 (마지막 조항은 부칙/별표 시작 전까지)
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else body_end
            content = text[start:end].strip()

            articles.append({
                "article_num": article_num,
                "article_sub": article_sub,
                "title": title,
                "content": content,
            })

    if buchik_match:
        articles.extend(_extract_buchik(text[body_end:attachment_start]))

    articles.extend(_extract_attachments(text[attachment_start:]))

    return articles


def _split_for_embedding(text: str, tokenizer, max_tokens: int) -> list[str]:
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
        chunk_overlap=min(20, max_tokens // 8),
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

    for article in articles:
        if "article_id" in article:
            # 부칙/별표/별지서식: extract_articles()가 이미 "부칙-1", "별표2" 같은
            # article_id를 만들어서 넘겨줌.
            article_id = article["article_id"]
        elif article["article_num"] == 0 and not article["title"]:
            # 본문에 "제XX조" 조항 헤더가 없는 경우(예: 별표 파일 하나가 통째로
            # 넘어온 경우). 파일명에서 조항 참조를 파싱할 수 있으면 그걸 쓰고,
            # 안 되면 "0" 유지.
            article_id = (_attachment_article_id(Path(source).name) if source else None) or "0"
        else:
            # 조항 번호 문자열 만들기 (예: "56", "56-2")
            article_id = str(article["article_num"])
            if article["article_sub"] is not None:
                article_id += f"-{article['article_sub']}"

        parent_records.append({
            "source": source,
            "article_id": article_id,
            "article_title": article["title"],
            "content": article["content"],
        })

        base_metadata = {
            **doc.metadata,   # 기존 metadata 유지
            "article_num": article.get("article_num"),
            "article_sub": article.get("article_sub"),
            "article_id": article_id,
            "article_title": article["title"],
        }

        if tokenizer is not None and max_tokens is not None:
            pieces = _split_for_embedding(article["content"], tokenizer, max_tokens)
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

    return split_law_document(combined_doc, tokenizer=tokenizer, max_tokens=max_tokens)