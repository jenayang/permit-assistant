"""프로젝트 전역 설정값.

환경 변수(.env)와 하드코딩된 기본값을 한 곳에서 관리합니다.
"""
from __future__ import annotations # 파이썬 신문법 사용가능(버전 상관없음)

import os
from pathlib import Path # 파일 경로 다루는 객체(깔끔, 편함)
import logging
from dotenv import load_dotenv

# .env 파일 로드(마크다운, PDF 등에서 API 키를 숨기기 위해 사용) -> .gitignore
load_dotenv()  # 모듈 로드 시점에 한 번만 실행
               # os.getenv()로 값을 꺼내려면 반드시 load_dotenv() 이후에 호출해야 함

# gRPC 로그 억제 (LLM 호출 시 나오는 FD 경고 숨김)
os.environ["GRPC_VERBOSITY"] = os.getenv("GRPC_VERBOSITY", "ERROR")
os.environ["GLOG_minloglevel"] = os.getenv("GLOG_minloglevel", "2")


# === 경로 ===
# 변수 생성 : 경로만 설정(아래 validate() mkdir()로 폴더 생성)
ROOT_DIR = Path(__file__).parent.parent   # __file__ : 현재 파일 경로, parent: 상위 폴더, parent.parent: 상위,상위 폴더(==프로젝트 루트)
DATA_DIR = ROOT_DIR / "data"              # '/' 연산자로 경로 결합
CHROMA_DIR = ROOT_DIR / "chroma_db"
EVAL_DIR = ROOT_DIR / "eval_results"


# === 모델 ===
# 다국어 지원 임베딩 모델 (한국어/영어 혼용 가능). 512토큰 한도.
# e5 계열은 입력 앞에 "query: "/"passage: " 접두사를 붙여야 성능이 나옴
# (retriever.py의 embeddings 생성부 참고).
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Gemini 모델 (무료 할당량이 가장 큰 모델)
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Cerebras 모델 (Gemini 무료 티어 일일 할당량 소진 시 자동 대체용, 한국어 지원)
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gemma-4-31b")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

# 브이월드(VWorld) - 주소로 용도지역 자동 조회(지오코더 + 2D데이터 API).
# 개발키는 지오코더만 즉시 열리고 2D데이터는 운영키 승인 후에 열림(2026-07-22 확인).
VWORLD_API_KEY = os.getenv("VWORLD_API_KEY")


# === 청킹 ===
# 임베딩 모델 자체 한도(예: 128토큰)가 이보다 작으면 모델 한도가 우선 적용됨.
# 모델을 큰 것(예: 512, 8192토큰)으로 바꿔도 청크가 무한정 커지지 않도록,
# 검색 정밀도 + LLM 컨텍스트 예산 관점에서 적절한 상한을 별도로 둔다.
MAX_PRACTICAL_CHUNK_TOKENS = 500

# 청크 경계끼리 겹치는 토큰 수(문맥 유실 완화용).
# 청크 자체가 이보다 작으면(예: 128토큰 모델) 비율로 축소해서 적용한다.
# 법령 자식 청크는 부모(조항 전체)가 문맥을 보장해줘서 적게 필요하고,
# 일반 문서는 청크 자체가 그대로 LLM에 노출되므로 더 크게 잡는다.
CHILD_CHUNK_OVERLAP_TOKENS = 20
GENERAL_CHUNK_OVERLAP_TOKENS = 50


# === 검색 ===
TOP_K = 3  # search_by_term(키워드 검색)에서 계속 사용

# search_regulations(벡터 검색)용: 1차로 넓게 후보를 뽑은 뒤(RERANK_CANDIDATE_K)
# CrossEncoder로 재정렬해서 상위 RERANK_TOP_K만 LLM에 넘긴다. 리랭킹이 의미
# 있으려면 후보 폭이 최종 개수보다 충분히 넉넉해야 한다.
RERANK_CANDIDATE_K = 15
RERANK_TOP_K = 5


# === ChromaDB ===
COLLECTION_NAME = "parmit-assistant"
DISTANCE_METRIC = "cosine"


# === 생성 ===
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0  # 100% 사실 기반
# gemini-2.5 계열은 내부 reasoning("thinking")도 MAX_OUTPUT_TOKENS를 소비함.
# 0으로 비활성화하지 않으면 예산이 작을 때 thinking만 하다 답변이 빈 문자열로 끝남.
THINKING_BUDGET = 0


# === LangSmith ===
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "permit-assistant")


# === LangGraph ===
MAX_HISTORY = 6


def validate() -> None:
    """필수 설정 검증 + 폴더 생성."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY가 설정되지 않았습니다.\n"
            ".env 파일에 키를 입력했는지 확인하세요.\n"
            "발급: https://aistudio.google.com/apikey"
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)   # 이때 폴더 생성
    CHROMA_DIR.mkdir(parents=True, exist_ok=True) # parents=True: 상위 폴더가 없으면 상위 폴더도 생성
    EVAL_DIR.mkdir(parents=True, exist_ok=True)   # exist_ok=True: 이미 폴더 있으면 그냥 넘어감(에러 안남)



def setup_logging(level: int = logging.INFO) -> None: # 로그 셋업(양식 설정)
    """프로젝트 전역 로깅 설정."""
    logging.basicConfig(
        level=level, # 디폴트 : INFO 이상 로그 출력
        format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d in %(funcName)s(): %(message)s", # 로그 출력 형식
        datefmt="%H:%M:%S", # asctime 형식 지정(디폴트: 2026-06-21 10:30:15,123) -> 10:30:15
    )

    # log level 5단계
    # DEBUG < INFO < WARNING < ERROR < CRITICAL

    # %s : string, %d : integer, %f : float, %x : hex, %o : octal
    # f-string"f{...}}"은 미리 변수가 정해져 있어야 실행됨.
    # logging은 나중에 값 채우짐 -> 지연 평가(lazy evaluation)
    # 나중에 logger.info(...)가 호출될 때, 실제로 로그 레벨이 활성화되어 있으면 문자열 포맷팅이 수행됨.