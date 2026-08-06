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
from src.pipeline import get_project_status, ingest, query, submit_facts, update_task_progress
from src.agent import ContextOverflowError
from src.rag.retriever import count_documents, find_unindexed_sources
from src.agents.permit import PROCEDURE_TREE, PermitResult

logger = logging.getLogger(__name__)


# === Lifecycle: 앱 시작 시 초기화 ===
def _warn_if_index_stale() -> None:
    """data/에 있는데 아직 인덱싱 안 된 파일이 있으면 시작 시 경고한다.

    인덱싱을 잊으면 검색이 에러를 내는 게 아니라 "가장 비슷한" 무관한 조문을
    조용히 돌려줘서, 답변은 그럴듯한데 근거만 틀린 상태가 된다 - 눈치채기
    가장 어려운 종류의 고장이라 시작할 때 눈에 띄게 알린다(2026-08-02에 실제로
    법령 6종이 누락된 채 운영되고 있었다, retriever.find_unindexed_sources 참고).

    경고만 하고 起動은 막지 않는다 - 일부 파일이 빠져도 나머지 기능은 정상이고,
    이 환경처럼 hwp5html이 없어 .hwp를 못 읽는 경우까지 서버를 못 뜨게 하면
    과한 조치다.
    """
    try:
        missing = find_unindexed_sources()
    except Exception as exc:  # DB가 아직 없을 수도 있음 - 점검 실패로 起動을 막지 않는다
        logger.warning("인덱스 최신성 점검을 건너뜁니다: %s", exc)
        return
    if not missing:
        return
    logger.warning(
        "인덱싱되지 않은 문서 %d건이 있습니다. 아직 인제스천을 안 돌렸다면 "
        "`uv run python -m src.pipeline --ingest`를 실행하세요 - 그래도 남는 파일은 "
        "추출할 본문이 없는 서식ᆞ이미지일 수 있고, 그 경우는 검색에서 빠지는 게 "
        "정상입니다(로그의 '→ 0 청크' 확인):\n%s",
        len(missing),
        "\n".join(f"  - {s}" for s in missing[:10])
        + (f"\n  ... 외 {len(missing) - 10}건" if len(missing) > 10 else ""),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # lifespan - 앱 생명주기 관리
    """앱 시작 시 설정 검증."""
    config.setup_logging()  # 로깅 설정
    config.validate()       # 검증(API키 확인)
    _warn_if_index_stale()
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


class LawContext(BaseModel):
    tool: str
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    user_id: str
    tool_calls: list[ToolCallInfo]
    permit_result: Optional[PermitResult] = None
    food_result: Optional[str] = None
    fire_result: Optional[list[str]] = None
    signage_result: Optional[str] = None
    task_progress: Optional[dict] = None
    project_status: Optional[dict] = None
    law_contexts: list[LawContext] = []


class TrackStatus(BaseModel):
    current_step: Optional[str] = None
    next_action: Optional[str] = None


class ProjectStatusResponse(BaseModel):
    progress: int
    completed: list[str]
    construction_track: TrackStatus
    startup_track: TrackStatus
    summary: str
    # 건축사 설계 의무 대상 여부(건축법 제23조). True=대행 필요, False=개인 직접
    # 가능, None=정보 부족. compute_project_status()가 규칙 엔진 결과를 그대로 실음.
    requires_architect: Optional[bool] = None


class TaskProgressUpdateRequest(BaseModel):
    user_id: str = Field(..., description="세션 ID")
    field: str = Field(..., description="task_progress 필드명(예: business_registration)")
    value: bool = Field(..., description="완료 여부(체크박스 상태)")


class TaskProgressUpdateResponse(BaseModel):
    task_progress: dict


class FactsSubmitRequest(BaseModel):
    """폼(intake.html) 제출용 - 채팅 없이 case_facts 등을 직접 채운다.

    각 *_facts는 대응하는 record_*_facts 도구(src/agents/*.py)와 동일한
    필드명을 그대로 받는 느슨한 dict다 - 도메인별 필드가 20개 가까이 돼서
    여기 각각을 다시 타입 선언하지 않는다(그 필드 계약의 단일 진실 공급원은
    record_*_facts 도구 시그니처).
    """
    user_id: str = Field(..., description="세션 ID")
    address: Optional[str] = Field(None, description="사업장 주소(입력 시 용도지역ᆞ건축물대장 자동조회)")
    case_facts: Optional[dict] = None
    food_facts: Optional[dict] = None
    fire_facts: Optional[dict] = None
    signage_facts: Optional[dict] = None


class FactsSubmitResponse(BaseModel):
    case_facts: dict
    food_facts: dict
    fire_facts: dict
    signage_facts: dict
    # build_permit_result()의 결정론적 미리보기(permit_type + procedures만) -
    # required_documents 등 LLM 종합이 필요한 필드는 비어 있다(finalize_node
    # 몫, 폼 흐름에는 안 걸림).
    permit_preview: Optional[PermitResult] = None
    food_result: Optional[str] = None
    fire_result: Optional[list[str]] = None
    signage_result: Optional[str] = None
    project_status: dict
    ledger_text: Optional[str] = Field(None, description="address 입력 시 조회된 건축물대장 원문(팝업 표시용)")


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
            signage_result=result["signage_result"],
            task_progress=result["task_progress"],
            project_status=result["project_status"],
            law_contexts=result["law_contexts"],
        )

    except ValueError as e: # 400 클라이언트 잘못(빈 질문 등)
        raise HTTPException(status_code=400, detail=str(e))
    except ContextOverflowError as e:
        # 프로바이더 원본 에러(영문 스택)를 그대로 노출하면 사용자가 무엇을
        # 해야 할지 알 수 없다. 지금까지의 판정 결과는 이미 상태에 남아 있으니
        # 새 대화로 이어가면 처음부터 다시 할 필요가 없다는 점을 안내한다.
        logger.error("컨텍스트 한도 초과: %s", e)
        raise HTTPException(
            status_code=413,
            detail="대화가 너무 길어져 한 번에 처리할 수 있는 분량을 넘었습니다. "
                   "새 대화를 시작해 주세요 - 지금까지 확인된 판정 결과는 "
                   "로드맵에 저장되어 있습니다.",
        )
    except Exception as e:  # 500 서버 잘못(LLM호출 실패 등)
        logger.error("질의 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"답변 생성 실패: {e}")


@app.get("/project-status", response_model=ProjectStatusResponse, tags=["로드맵"])
def project_status_endpoint(user_id: str) -> ProjectStatusResponse:
    """대화 없이(페이지 로드 시점 등) 현재 프로젝트 진행 현황만 조회한다.

    재접속 요약 배너ᆞ사이드바 Next Action 위젯이 이 엔드포인트를 쓴다 -
    LLM 호출 없이 그래프 상태에서 바로 계산(project_status.py 참고).
    """
    try:
        return ProjectStatusResponse(**get_project_status(user_id))
    except Exception as e:
        logger.error("진행 현황 조회 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"진행 현황 조회 실패: {e}")


@app.post("/task-progress", response_model=TaskProgressUpdateResponse, tags=["로드맵"])
def task_progress_endpoint(req: TaskProgressUpdateRequest) -> TaskProgressUpdateResponse:
    """로드맵 체크박스를 사용자가 직접 클릭했을 때 진행상황을 갱신한다.

    /query(대화)를 거치지 않고 그래프 상태를 바로 patch한다(LLM 호출 없음) -
    체크박스 클릭도 record_task_progress와 신뢰 수준이 같은 사용자 자기보고라
    자연어를 왕복시킬 필요가 없다.
    """
    try:
        task_progress = update_task_progress(req.user_id, req.field, req.value)
        return TaskProgressUpdateResponse(task_progress=task_progress)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("진행상황 갱신 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"진행상황 갱신 실패: {e}")


@app.post("/facts", response_model=FactsSubmitResponse, tags=["로드맵"])
def facts_endpoint(req: FactsSubmitRequest) -> FactsSubmitResponse:
    """intake.html 폼 제출 - 채팅 없이 case_facts 등을 직접 반영하고 판정한다.

    /task-progress와 같은 원칙(LLM 호출 없음, 그래프 상태를 직접 patch) -
    다만 이쪽은 patch 후 classify_node까지 재사용해 판정 결과도 함께 낸다.
    """
    try:
        result = submit_facts(
            user_id=req.user_id,
            address=req.address,
            case_facts=req.case_facts,
            food_facts=req.food_facts,
            fire_facts=req.fire_facts,
            signage_facts=req.signage_facts,
        )
        return FactsSubmitResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("폼 제출 처리 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"폼 제출 처리 실패: {e}")


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
