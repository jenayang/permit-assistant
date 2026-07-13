"""문서 로딩 + 전처리 + Split.

지원 포맷: pdf, md, txt, html
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 

from src import config
from src.law_chunker import is_law_document, split_law_documents

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

LOADERS = {
    ".pdf": _load_pdf,
    ".md": _load_text,
    ".txt": _load_text,
    ".html": _load_html,
    ".htm": _load_html,
}
# 지원하는 확장자 목록 
SUPPORTED_EXTENSIONS = set(LOADERS.keys()) 


def load_document(filepath: Path) -> list[Document]:
    ext = filepath.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(
            f"지원하지 않는 형식 : {ext}"
            f"지원: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return LOADERS[ext](filepath) # LOADERS[.pdf] -> _load_pdf(filepath) 형태로 호출


# === 전처리 ===(# 2. 전처리)
def preprocess_text(text: str) -> str:
    """유니코드 정규화 + 제어 문자 제거 + 공백 정리."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\xad]", "", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess_documents(docs: list[Document]) -> list[Document]:
    """Document 리스트의 page_content를 정제."""
    for d in docs:
        d.page_content = preprocess_text(d.page_content)
    return docs


# === 통합 처리 ===(나. 파일로드, 전처리, 청킹)
def process_file(filepath: Path) -> list[Document]:
    """파일 하나를 로드 → 전처리 → 청킹 """
    # 1. 파일 로딩 -> list[Document]
    docs = load_document(filepath)
    # 2. 전처리(각 Document의 .page_context 갈아끼기) -> list[Document]
    docs = preprocess_documents(docs)
    
    # 2.5. metadata 보강
    # loader마다 metadata 형식이 달라서, 통일성 위해 명시적으로 설정
    try:
        relative_path = str(filepath.relative_to(config.DATA_DIR))
    except ValueError:
        relative_path = filepath.name

    for d in docs:
        d.metadata["source"] = relative_path            # "laws/건축법.pdf"
        d.metadata["category"] = filepath.parent.name   # "laws"
        d.metadata["format"] = filepath.suffix.lstrip(".")

    # 3. Split - 법령이면 조항 단위, 아니면 일반 청킹
    if is_law_document(filepath):
        logger.info("법령 청킹 적용: %s", filepath.name)
        split_docs = split_law_documents(docs)
    else:
        splitter = RecursiveCharacterTextSplitter( 
            chunk_size = config.CHUNK_SIZE,
            chunk_overlap = config.CHUNK_OVERLAP
        )
        split_docs = splitter.split_documents(docs) # Document 리스트 분할
    
    # 4. chunk_index 추가 (디버깅과 검색에 유용)
    for i, c in enumerate(split_docs):
        c.metadata["chunk_index"] = i
    
    return split_docs


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


def ingest_all(data_dir: Path = config.DATA_DIR) -> list[Document]: # ****
    """data 폴더 전체를 청크로 변환."""
    files = discover_files(data_dir) # (가.모든 파일 찾기 -> files[] 리스트)
    if not files:
        logger.warning("처리할 파일이 없습니다: %s", data_dir)
        return []

    all_docs: list[Document] = [] # 청킹된 모든 파일
    for filepath in files: # 불러온 파일 하나씩 처리
        try:
            split_docs = process_file(filepath) # (나. 파일 로딩 > 전처리 > 청킹)
            all_docs.extend(split_docs)       # 각 파일의 청크들 -> all_청크에 append
            logger.info("[완료] %s → %d 청크", filepath.name, len(split_docs))
        except Exception as e:
            logger.error("[실패] %s: %s", filepath.name, e)

    return all_docs # 전체 파일의 총 청크
