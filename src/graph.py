"""LangGraph 그래프 조립 (agent.py에서 분리, Phase 2).

이 모듈이 agent.py의 노드ᆞ라우팅 함수를 가져다 배선만 한다 - 반대 방향
(agent.py가 이 모듈을 import)은 순환참조가 되므로 금지. `graph`(컴파일된
그래프 인스턴스)를 쓰는 곳(pipeline.py 등)은 이 모듈에서 바로 가져올 것.

그래프 구조:
    START → agent → [도구 필요?]
                    ├─ 실제 도구 호출 있음 → tools → agent (순환)
                    ├─ case_facts 다 모였고 아직 미분류 → classify → agent (한 번 더,
                    │  분류 결과를 답변에 자연스럽게 반영하도록)
                    └─ 자유 텍스트 답변 → guard → [위반?]
                                                 ├─ 있음(재시도 1회 이내) → agent
                                                 ├─ 없음+분류 끝났고 미종합 → finalize → END
                                                 └─ 없음+그 외 → END
"""
from __future__ import annotations

import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src import config
from src.agent import AgentState, agent_node, finalize_node, tool_node
from src.agents.permit import PermitResult
from src.classify import classify_node
from src.guard import guard_node
from src.routing import route_after_agent, route_after_guard, route_after_tools


def build_graph():
    """ReAct 에이전트 그래프 구성. checkpointer가 user_id(=thread_id)별 대화
    이력을 SQLite에 남긴다."""
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("classify", classify_node)
    builder.add_node("guard", guard_node)
    builder.add_node("finalize", finalize_node)

    # path_map(3번째 인자)은 생략하면 안 된다 - 없으면 LangGraph가 분기 대상을
    # 정적으로 알 수 없어 get_graph()가 조건부 엣지를 전부 잃고 "agent → END"
    # 하나로 뭉갠다(2026-08-03 실측). 런타임 라우팅은 되지만 그래프 구조 조회ᆞ
    # 시각화가 망가진다.
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "guard": "guard",
        }
    )
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "classify": "classify",  # case_facts가 방금 완성됨 - agent가 답하기 전에 먼저 분류
            "agent": "agent",        # 순환엣지 (기존과 동일)
        }
    )
    builder.add_edge("classify", "agent")   # 분류 결과를 LLM이 답변에 반영하도록 한 번 더
    builder.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            "agent": "agent",        # 위반 감지 - 재시도 유도 메시지를 보고 다시 답변
            "finalize": "finalize",
            END: END,
        }
    )
    builder.add_edge("finalize", END)       # LLM 재호출 없이 그대로 종료

    # 체크포인터로 대화 이력 관리. SQLite에 저장해서 서버 재시작에도
    # user_id(=thread_id)별 대화ᆞcase_facts가 유지되도록 한다(예전엔 InMemorySaver라
    # 재시작하면 전부 소실됐음). check_same_thread=False로 여는 이유: FastAPI가
    # 동기 라우트를 스레드풀에서 돌려서 요청마다 다른 스레드가 이 커넥션을 쓸 수
    # 있는데, SqliteSaver 내부에 threading.Lock이 있어 동시 접근이 안전하다.
    # parent_store.py와 같은 폴더(CHROMA_DIR)에 저장 - 이 프로젝트의 기존 SQLite
    # 파일 위치 관례를 따름.
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.CHROMA_DIR / "checkpoints.sqlite"), check_same_thread=False)
    # 체크포인터는 상태(AgentState)를 msgpack으로 직렬화하는데, permit_result가
    # 커스텀 Pydantic 모델(PermitResult)이라 기본 직렬화기가 "등록 안 된 타입"
    # 경고를 냄 - 지금은 허용하되 경고만 남기는 완화 모드지만, 미래 LangGraph
    # 버전에서 기본이 차단으로 바뀔 예정이라 명시적으로 허용 목록에 등록해둔다
    # (전체 허용(allowed_msgpack_modules=True) 대신 이 타입 하나만 등록 - 불필요한
    # 타입까지 역직렬화 허용하지 않도록).
    serde = JsonPlusSerializer(allowed_msgpack_modules=[PermitResult])
    checkpointer = SqliteSaver(conn, serde=serde)  # short-term memory (영구 저장)
    checkpointer.setup()
    return builder.compile(checkpointer=checkpointer)


# === 컴파일된 그래프 (모듈 로드 시 1회) ===
graph = build_graph()
