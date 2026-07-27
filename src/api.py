"""FastAPI 서버.

실행:
    uvicorn src.api:app --reload

API 문서: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.pipeline import ingest, query
from src.retriever import count_documents
from src.agents.permit import PROCEDURE_TREE, PermitResult

logger = logging.getLogger(__name__)


# === Lifecycle: 앱 시작 시 초기화 ===
@asynccontextmanager
async def lifespan(app: FastAPI):  # lifespan - 앱 생명주기 관리
    """앱 시작 시 설정 검증."""
    config.setup_logging()  # 로깅 설정
    config.validate()       # 검증(API키 확인)
    logger.info("RAG API 시작 완료")
    yield
    logger.info("RAG API 종료")


# === FastAPI 앱 ===
app = FastAPI(
    title="permit-assistant",
    description="건축 인허가 어시스턴스",
    version="0.1.0",
    lifespan=lifespan
)


# === Pydantic 모델(요청/응답 스키마) ===
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="질문")
    user_id: Optional[str] = Field(None, description="대화 이력 유지용 세션 ID")


class ToolCallInfo(BaseModel):
    tool: str
    args: dict


class QueryResponse(BaseModel):
    question: str
    answer: str
    user_id: str
    tool_calls: list[ToolCallInfo]
    permit_result: Optional[PermitResult] = None
    food_result: Optional[str] = None
    fire_result: Optional[list[str]] = None


class IngestResponse(BaseModel):
    ingested_chunks: int
    total_chunks_in_db: int


class HealthResponse(BaseModel):
    status: str
    chunks_in_db: int
    version: str



# === 엔드포인트 ===
@app.get("/", tags=['기본'])
def root():
    return {
        "service": "RAG Chatbot API v3 (LangGraph)",
        "version": "3.0.0",
        "docs": "/docs",
        "ui": "/ui",
    }


# 정적 프론트엔드 (채팅 UI): http://localhost:8000/ui
app.mount("/ui", StaticFiles(directory="src/static", html=True), name="ui")


@app.get("/health", response_model=HealthResponse, tags=["기본"])
def health() -> HealthResponse:
    """서버 상태 확인."""
    return HealthResponse(
        status="ok",
        chunks_in_db=count_documents(),
        version="3.0.0"
    )


@app.get("/procedure-stages", tags=["기본"])
def procedure_stages() -> dict:
    """케이스 유형별 절차 트리 (분기 포함, 프론트 로드맵/버튼 UI용, 시작 시 1회 캐싱)."""
    return PROCEDURE_TREE


@app.post("/query", response_model=QueryResponse, tags=["질의응답"])
def query_endpoint(req: QueryRequest) -> QueryResponse:
    """질문에 답변.
    
    같은 user_id로 요청 시 이전 대화 맥락 유지.
    """
    try:
        result = query(req.question, user_id=req.user_id) # {"answer": "...", "sources": [...]}
        return QueryResponse(
            question=req.question,
            answer=result["answer"],
            user_id=result["user_id"],
            tool_calls=result["tool_calls"],
            permit_result=result["permit_result"],
            food_result=result["food_result"],
            fire_result=result["fire_result"],
        )
        
    except ValueError as e: # 400 클라이언트 잘못(빈 질문 등)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # 500 서버 잘못(LLM호출 실패 등)
        logger.error("질의 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"답변 생성 실패: {e}")


@app.post("/ingest", response_model=IngestResponse, tags=["인덱싱"])
def ingest_endpoint(reset: bool = False) -> IngestResponse:
    """data 폴더의 문서들을 인덱싱."""
    try:
        n = ingest(reset=reset)
        return IngestResponse(
            ingested_chunks=n,
            total_chunks_in_db=count_documents(),
        )
    except Exception as e:
        logger.exception("인덱싱 실패: %s, e", exc_info=True)
        raise HTTPException(status_code=500, detail=f"인덱싱 실패: e")
