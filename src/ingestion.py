"""문서 로딩 + 전처리 + Split.

지원 포맷: pdf, md, txt, html, hwp
"""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 

from src import config
from src.law_chunker import is_law_document, split_law_documents
from src.retriever import get_max_chunk_tokens, get_tokenizer

logger = logging.getLogger(__name__)  # "src.ingestion" 이름으로 로거 생성


# === 데이터 클래스 === (삭제)


# === 로더 === (# 1. 파일 로딩)
def _load_pdf(filepath: Path) -> list[Document]:
    return PyMuPDFLoader(str(filepath)).load() 

def _load_text(filepath: Path) -> list[Document]:
    """md, txt 공통."""
    return TextLoader(str(filepath), encoding="utf-8").load()

def _load_html(filepath: Path) -> list[Document]:
    return UnstructuredHTMLLoader(str(filepath)).load()
    # WebBaseLoader은 로컬 HTML파일엔 안맞음(.html파일 -> Unstructured..)

def _load_hwp(filepath: Path) -> list[Document]:
    """hwp5txt는 표(별표) 안의 텍스트를 못 읽어서(HWPTAG_TABLE을 안 따라감),
    hwp5html로 표까지 <table>로 변환한 뒤 기존 HTML 로더로 읽는다.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        subprocess.run(
            ["hwp5html", f"--output={out_dir}", str(filepath)],
            check=True, capture_output=True, text=True,
        )
        return _load_html(out_dir / "index.xhtml")

LOADERS = {
    ".pdf": _load_pdf,
    ".md": _load_text,
    ".txt": _load_text,
    ".html": _load_html,
    ".htm": _load_html,
    ".hwp": _load_hwp,
}
# 지원하는 확장자 목록 
SUPPORTED_EXTENSIONS = set(LOADERS.keys()) # 중복삭제


def load_document(filepath: Path) -> list[Document]:
    ext = filepath.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(
            f"지원하지 않는 형식 : {ext}"
            f"지원: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return LOADERS[ext](filepath) # LOADERS[.pdf] -> _load_pdf(filepath) 형태로 호출


# 조문 중간에 끼는 개정 이력 표기(예: "<개정 2009. 7. 16., 2010. 2. 18.>").
# 키워드로 한정해서, 부칙 표제의 공포번호 인용("부칙 <제35811호, 2025. 10. 1.>")은
# 안 건드리게 한다(law_chunker의 부칙 구역 감지가 이 "<"에 의존함).
REVISION_HISTORY_PATTERN = re.compile(r"<(?:개정|신설|전문개정|일부개정|삭제|생략)[^<>]*>")

# 삭제/생략된 조항 표기(예: "제42조 삭제 <1999. 4. 30.>", "제23조부터 제23조의8까지 삭제").
# ARTICLE_PATTERN이 제목 없는 조항은 헤더로 안 잡아서 앞 조항 내용에 조용히
# 흡수는 되는데, 텍스트 자체는 안 지워지고 그대로 남아있어서 별도로 제거한다.
REPEALED_STUB_PATTERN = re.compile(
    r"(?:^|\n)\s*제\s*\d+\s*조(?:의\s*\d+)?"
    r"(?:\s*(?:부터|내지)\s*제\s*\d+\s*조(?:의\s*\d+)?\s*까지)?"
    r"\s*(?:삭제|생략)\s*(?:<[^<>]*>)?\s*(?=\n|$)"
)


# === 전처리 ===(# 2. 전처리)
def preprocess_text(text: str) -> str:
    """유니코드 정규화 + 제어 문자 제거 + 개정 이력/삭제 조항 표기 제거 + 공백 정리."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\xad]", "", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)
    text = REVISION_HISTORY_PATTERN.sub("", text)
    text = REPEALED_STUB_PATTERN.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess_documents(docs: list[Document]) -> list[Document]:
    """Document 리스트의 page_content를 정제."""
    for d in docs:
        d.page_content = preprocess_text(d.page_content)
    return docs


# === 통합 처리 ===(나. 파일로드, 전처리, 청킹)
def process_file(filepath: Path) -> tuple[list[Document], list[dict]]:
    """파일 하나를 로드 → 전처리 → 청킹.

    Returns:
        (청크 리스트, 부모 조항 레코드 리스트). 법령 문서가 아니면 부모
        레코드는 빈 리스트.
    """
    # 1. 파일 로딩 -> list[Document]
    docs = load_document(filepath)
    # 2. 전처리(각 Document의 .page_context 갈아끼기) -> list[Document]
    docs = preprocess_documents(docs)

    # 2.5. metadata 보강
    # loader마다 metadata 형식이 달라서, 통일성 위해 명시적으로 설정
    try:
        relative_path = str(filepath.relative_to(config.DATA_DIR)) # "data"
        # relative_to : 절대 경로(/Users/.../data/laws/건축법.pdf)를 -> 상대 경로(laws/건축법.pdf)로 변환
        # 이 filepath(==객체)를 순수 str 문자열로 변환  -> metadata["source"]에 str으로 넣기 위함(객체는 json 직렬화 불가)
    except ValueError:
        relative_path = filepath.name

    for d in docs:
        d.metadata["source"] = relative_path            # "laws/건축법.pdf"(str)
        d.metadata["category"] = filepath.parent.name   # "laws"
        d.metadata["format"] = filepath.suffix.lstrip(".") # 확장자(ex. pdf)
        # 키 값이 있으면 덮어쓰기, 없으면 새로 추가 (로더의 원 메타데이터 값이랑 상관없이, 우리가 원하는 메타데이터로 만들기)
        # PDF(PyMuPDFLoader) : metadata["format"]으로는 pdf의 버전 들어가있음 -> 이걸 우리가 원하는 값(확장자명)으로 덮어쓰기

    # 3. Split - 법령이면 조항 단위(+ 토큰 한도 초과 시 서브 분할), 아니면 일반 청킹
    #            (법령: 조항==부모 청킹, 모델의 최대 토큰 넘어가면 서브청킹==자식 청킹)
    parent_records: list[dict] = []
    if is_law_document(filepath):
        logger.info("법령 청킹 적용: %s", filepath.name)
        split_docs, parent_records = split_law_documents(
            docs, tokenizer=get_tokenizer(), max_tokens=get_max_chunk_tokens()
        )
    else:
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            get_tokenizer(),
            chunk_size=get_max_chunk_tokens(),
            chunk_overlap=config.GENERAL_CHUNK_OVERLAP_TOKENS,
        )
        split_docs = splitter.split_documents(docs) # Document 리스트 분할

    # 4. chunk_index 추가 (디버깅과 검색에 유용)
    for i, c in enumerate(split_docs):
        c.metadata["chunk_index"] = i

    return split_docs, parent_records


# (가. 모든 파일 담기)
def discover_files(data_dir: Path = config.DATA_DIR) -> list[Path]:
    """data 폴더에서 지원하는 파일 모두 찾기."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(data_dir.rglob(f"*{ext}")) # 각 확장자에 해당하는 파일 찾아서 files 리스트에 추가(rglob 재귀적 찾기)

    files = [
        f for f in files
        if not f.name.lower().startswith("readme")
    ]
    return sorted(files)


def ingest_all(data_dir: Path = config.DATA_DIR) -> tuple[list[Document], list[dict]]:
    """data 폴더 전체를 청크로 변환.

    Returns:
        (전체 청크 리스트, 전체 부모 조항 레코드 리스트).
    """
    files = discover_files(data_dir) # (가.모든 파일 찾기 -> files[] 리스트)
    if not files:
        logger.warning("처리할 파일이 없습니다: %s", data_dir)
        return [], []

    all_docs: list[Document] = [] # 청킹된 모든 파일
    all_parent_records: list[dict] = []
    for filepath in files: # 불러온 파일 하나씩 처리
        try:
            split_docs, parent_records = process_file(filepath) # (나. 파일 로딩 > 전처리 > 청킹)
            all_docs.extend(split_docs)       # 각 파일의 청크들 -> all_청크에 append
            all_parent_records.extend(parent_records)
            logger.info("[완료] %s → %d 청크", filepath.name, len(split_docs))
        except Exception as e:
            logger.error("[실패] %s: %s", filepath.name, e)

    return all_docs, all_parent_records # 전체 파일의 총 청크 + 부모 레코드
