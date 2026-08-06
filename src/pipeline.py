"""전체 RAG 파이프라인 통합 + CLI.

사용법:
    python -m src.pipeline --ingest                   # 인덱싱
    python -m src.pipeline --ingest --reset           # 인덱싱 초기화
    python -m src.pipeline --query "질문"              # 질문
    python -m src.pipeline --interactive 
    python -m src.pipeline --interactive --user_id "유저 이름"

"""
from __future__ import annotations

import argparse
import logging
import uuid
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage

from src import config
from src.agent import extract_text, merge_facts
from src.agents.permit import build_permit_result
from src.agents.site import get_land_zone_category, lookup_building_ledger
from src.classify import classify_node
from src.graph import graph
from src.guard import GROUNDING_TOOL_NAMES
from src.agents.roadmap_progress import TASK_PROGRESS_FIELDS, cascade_construction_progress
from src.rag.ingestion import ingest_all
from src.project_status import compute_project_status
from src.rag.retriever import index_documents, count_documents, reset_collection
from src.rag import parent_store

logger = logging.getLogger(__name__)


# === 인덱싱 ===(만들기+저장 -> 반환: 요약정보)
def ingest(reset: bool = False) -> int:
    """data 폴더의 모든 문서를 인덱싱."""
    if reset: # reset=True면 기존 컬렉션 초기화 후 인덱싱
        logger.info("기존 컬렉션 초기화 중...")
        reset_collection()

    docs, parent_records = ingest_all()

    if not docs:
        return 0

    logger.info("문서 로딩 + 청킹 완료, 증분 인덱싱 시작")
    result = index_documents(docs)
    parent_store.replace_all(parent_records)

    total = count_documents()
    logger.info(
        "인덱싱 완료: 추가 %d / 갱신 %d / 스킵 %d / 삭제 %d (DB 총 %d개)",
        result["num_added"], result["num_updated"], result["num_skipped"], result["num_deleted"], total,
    )
    return result["num_added"] + result["num_updated"] # 실제로 새로 (재)임베딩된 청크 수

# [INFO] src.pipeline: 기존 컬렉션 초기화 중...
# [INFO] src.retriever: 기존 컬렉션 삭제 완료

# [INFO] src.pipeline: 문서 로딩 + 청킹 시작
# [INFO] src.ingestion: [완료] sample_pdf.pdf → 5 청크
# [INFO] src.ingestion: [완료] sample_homepage.html → 8 청크
# [INFO] src.ingestion: [완료] sample_reviews.json → 3 청크
# [INFO] src.ingestion: [완료] sample_textbook.md → 11 청크
# [INFO] src.pipeline: 인덱싱 완료: 27개 청크 (DB 총 27개)


# === 질의응답 ===
def query(question: str, user_id: Optional[str] = None) -> dict:
    """질문에 답변 생성.
        Args:
            question: 사용자 질문
            user_id: 사용자 식별자 (같은 user_id는 대화 이력 공유)
    
    Returns:
        {"answer": ..., "user_id": ..., "tool_calls": [...]}
    """
    if not question.strip():
        raise ValueError("질문이 비어 있습니다.")
    
    if user_id is None:
        user_id = str(uuid.uuid4())

    config_dict = {"configurable": {"thread_id": user_id}}

    logger.info("[User %s] 질문: %s", user_id, question)


    # 그래프 실행
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config = config_dict
    )

    # 마지막 AI 메시지 추출 (content가 리스트인 최신 Gemini 응답 형식도
    # extract_text가 처리 - finalize_node의 explanation 추출과 동일 로직 공유)
    answer = extract_text(result["messages"][-1].content)

    # result["messages"]는 체크포인터가 대화 시작부터 지금까지 누적한 전체
    # 이력이라, 그대로 훑으면 이전 턴에 불렀던 도구까지 이번 턴 결과에 계속
    # 같이 잡힌다(실제로 겪음 - 대화가 길어질수록 도구 호출 목록이 끝없이
    # 누적되어 보임). 이번 턴은 마지막 HumanMessage(방금 질문) 이후부터라서,
    # 그 지점부터만 잘라서 본다.
    last_human_idx = max(
        i for i, msg in enumerate(result["messages"]) if isinstance(msg, HumanMessage)
    )
    current_turn_messages = result["messages"][last_human_idx:]

    # 도구 호출 이력 추출 (이번 턴만)
    tool_calls = []
    for msg in current_turn_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:  # msg.tool_calls: 실제 내용이 들어있는지 확인.(AIMessage는 항상 있음 [])
            for tc in msg.tool_calls:
                tool_calls.append({
                    "tool": tc["name"],
                    "args": tc["args"],
                })

    # 검색된 컨텍스트 추출 (이번 턴만 - RAGAS 등 평가에 사용, 실제로 LLM에 들어간 근거 텍스트)
    contexts = []
    for msg in current_turn_messages:
        if isinstance(msg, ToolMessage):
            contexts.extend(part for part in msg.content.split("\n\n") if part.strip())

    # "법령 원문 보기" 카드용 - contexts와 달리 record_case_facts("기록됨.")나
    # lookup_building_ledger(건축물대장 조회 결과) 같은 비-법령 도구 응답은
    # 빼고, 실제로 [출처: ...] 인용의 근거가 되는 도구(GROUNDING_TOOL_NAMES,
    # guard_node와 동일 기준)만 남긴다 - 안 그러면 카드에 "기록됨." 같은
    # 엉뚱한 내용이 법령 원문인 것처럼 뜬다.
    law_contexts = []
    for msg in current_turn_messages:
        if isinstance(msg, ToolMessage) and msg.name in GROUNDING_TOOL_NAMES:
            law_contexts.append({"tool": msg.name, "text": msg.content})

    logger.info("[User: %s] 답변 길이: %d, 도구 호출: %d회",
                user_id, len(answer), len(tool_calls))

    return {
        "answer": answer,
        "user_id": user_id,
        "tool_calls": tool_calls,
        "contexts": contexts,
        "law_contexts": law_contexts,
        "permit_result": result.get("permit_result"),
        "food_result": result.get("food_result"),
        "fire_result": result.get("fire_result"),
        "signage_result": result.get("signage_result"),
        "task_progress": result.get("task_progress"),
        "project_status": compute_project_status(result),
    }


# === 프로젝트 진행 현황 조회 (LLM 호출 없음) ===
def get_project_status(user_id: str) -> dict:
    """재접속 요약 배너ᆞNext Action 위젯 등이 페이지 로드 시(대화 없이도)
    바로 쓸 수 있도록 현재 그래프 상태에서 진행 현황만 조회한다. query()의
    project_status와 같은 compute_project_status()를 재사용해 두 경로가
    서로 다른 계산을 하지 않게 한다.
    """
    config_dict = {"configurable": {"thread_id": user_id}}
    state = graph.get_state(config_dict)
    return compute_project_status(state.values)


# === 로드맵 체크박스 직접 갱신 (LLM 호출 없음) ===
def update_task_progress(user_id: str, field: str, value: bool) -> dict:
    """사용자가 프론트 로드맵 체크박스를 직접 클릭했을 때 task_progress를
    그래프 상태에 바로 반영한다. record_task_progress(LLM 도구)가 대화 중
    자연어로 자기보고를 받는 것과 신뢰 수준이 동일한 자기보고라 - 체크박스
    클릭도 "사용자가 방금 완료라고 밝힌 것"이므로 - 굳이 LLM을 거쳐 자연어를
    왕복시키지 않고 graph.update_state로 직접 패치한다(대화 이력에는 안
    남지만 다음 질의ᆞ답변에는 그대로 반영됨).
    """
    if field not in TASK_PROGRESS_FIELDS:
        raise ValueError(f"알 수 없는 진행상황 항목: {field}")

    config_dict = {"configurable": {"thread_id": user_id}}
    patch = cascade_construction_progress({field: value})
    graph.update_state(config_dict, {"task_progress": patch})
    state = graph.get_state(config_dict)
    return state.values.get("task_progress") or {}


# === 폼 기반 판정 (LLM 호출 없음) ===
def submit_facts(
    user_id: str,
    address: Optional[str] = None,
    case_facts: Optional[dict] = None,
    food_facts: Optional[dict] = None,
    fire_facts: Optional[dict] = None,
    signage_facts: Optional[dict] = None,
) -> dict:
    """채팅 대화 없이(폼 제출) case_facts 등을 그래프 상태에 직접 반영하고
    판정까지 돌린다. update_task_progress와 같은 원칙(LLM 미호출, 결과가
    이미 그래프에 있는 순수 함수 classify_node/build_permit_result 재사용) -
    실제 대화 중 tools→classify로 이어지는 그래프 흐름을 그대로 두 단계로
    수동 재현한다.
    """
    config_dict = {"configurable": {"thread_id": user_id}}

    # 1단계: 사용자가 제출한 facts를 먼저 반영(merge_facts 리듀서가 기존
    # 상태와 병합) - record_case_facts 등 도구가 하는 일과 동일한 patch.
    patch: dict = {}
    if case_facts:
        patch["case_facts"] = case_facts
    if food_facts:
        patch["food_facts"] = food_facts
    if fire_facts:
        patch["fire_facts"] = fire_facts
    if signage_facts:
        patch["signage_facts"] = signage_facts
    if patch:
        graph.update_state(config_dict, patch)

    state = graph.get_state(config_dict).values

    # 이번 턴에 새로 남길 메시지는 반드시 HumanMessage로 시작해야 한다 - 폼만
    # 제출하고 채팅을 한 번도 안 한 세션에서 이력의 첫 메시지가 곧바로
    # AIMessage(tool_calls)로 남으면, 이후 실제 채팅에서 Gemini가 "함수 호출
    # 턴은 사용자 턴 또는 함수 응답 턴 뒤에만 올 수 있다"는 제약을 위반했다며
    # 400 INVALID_ARGUMENT로 거부하는 걸 실측 확인(2026-08-04) - 실제 대화는
    # 항상 HumanMessage로 시작해서 이 문제가 없었다.
    turn_messages: list = []
    if address or patch:
        summary = []
        if address:
            summary.append(f"주소 {address}")
        for label, facts in (("건축", case_facts), ("식품위생", food_facts), ("소방", fire_facts), ("간판", signage_facts)):
            if facts:
                summary.append(f"{label} 정보 {len(facts)}건")
        turn_messages.append(HumanMessage(content="[폼 제출] " + ", ".join(summary)))

    # case_facts/food_facts/fire_facts/signage_facts 각각을 실제 record_*_facts
    # 호출처럼 tool_calls/ToolMessage 쌍으로 이력에 남긴다 - 위 HumanMessage는
    # "건축 정보 2건"처럼 개수만 말해서, 어떤 필드가 무슨 값으로 기록됐는지는
    # LLM이 대화 이력 어디서도 볼 수 없었다(2026-08-06 재신고: intake 폼으로
    # ownership="소유자"를 넣었는데 챗봇이 소유자/임차인을 다시 물어봄 - 실제
    # 채팅 중 LLM이 직접 record_case_facts를 부르면 그 tool_calls 자체가
    # 이력에 남아 다음 턴에 "이미 물어봤다"는 걸 알 수 있는데, 폼 제출은 이
    # 흔적이 없어서 항상 재질문으로 이어졌다). lookup_building_ledger가 이미
    # 쓰는 것과 같은 패턴 - args에 실제 값이 그대로 담기므로 별도 텍스트
    # 요약이 필요 없다.
    for tool_name, facts in (
        ("record_case_facts", case_facts),
        ("record_food_facts", food_facts),
        ("record_fire_facts", fire_facts),
        ("record_signage_facts", signage_facts),
    ):
        if not facts:
            continue
        call_id = f"formrecord_{uuid.uuid4().hex[:8]}"
        turn_messages.append(AIMessage(content="", tool_calls=[{"name": tool_name, "args": facts, "id": call_id}]))
        turn_messages.append(ToolMessage(content="기록됨.", tool_call_id=call_id, name=tool_name))

    # 주소가 있으면 기존 site.py 조회 도구를 그대로 재사용해 용도지역ᆞ
    # 건축물대장을 자동으로 채운다(사용자가 이미 입력한 값은 덮지 않음).
    ledger_text: Optional[str] = None
    if address:
        zone = get_land_zone_category(address)
        if zone and not (state.get("case_facts") or {}).get("land_zone"):
            graph.update_state(config_dict, {"case_facts": {"land_zone": zone}})

        ledger_text = lookup_building_ledger.invoke({"address": address})
        call_id = f"formlookup_{uuid.uuid4().hex[:8]}"
        turn_messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "lookup_building_ledger", "args": {"address": address}, "id": call_id}],
            )
        )
        turn_messages.append(ToolMessage(content=ledger_text, tool_call_id=call_id, name="lookup_building_ledger"))

    if turn_messages:
        graph.update_state(config_dict, {"messages": turn_messages})

    # 2단계: classify_node를 직접 호출(실제 그래프의 tools→classify와 동일
    # 절차) - 방금 반영한 메시지ᆞfacts를 포함한 최신 state를 다시 읽어서
    # 넘긴다(_derive_ledger_facility_group이 방금 persist한 lookup_building_ledger
    # ToolMessage를 실제로 찾아야 하므로).
    state = graph.get_state(config_dict).values
    # auto_search=False: 폼 제출은 응답을 즉시 돌려줘야 해서 판정 직후 rerank
    # 검색을 생략한다(classify_node/run_classifier 주석 참고, 2026-08-05 피드백 -
    # intake 폼에서 여러 도메인이 한 번에 판정될 때 rerank가 순차로 겹쳐 느렸음).
    classify_update = classify_node(state, auto_search=False)
    if classify_update:
        graph.update_state(config_dict, classify_update)

    final = graph.get_state(config_dict).values
    return {
        "case_facts": final.get("case_facts") or {},
        "food_facts": final.get("food_facts") or {},
        "fire_facts": final.get("fire_facts") or {},
        "signage_facts": final.get("signage_facts") or {},
        "permit_preview": build_permit_result(final.get("case_facts") or {}),
        "food_result": final.get("food_result"),
        "fire_result": final.get("fire_result"),
        "signage_result": final.get("signage_result"),
        "project_status": compute_project_status(final),
        "ledger_text": ledger_text,
    }


# === CLI ===
def main() -> None:
    # argparse로 명령행 인자 정의
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI (v3 langGraph)")
    parser.add_argument("--ingest", action="store_true", help="data/ 폴더 인덱싱")
    parser.add_argument("--reset", action="store_true", help="인덱싱 전 DB 초기화")
    parser.add_argument("--query", type=str, default=None, help="단발 질문")
    parser.add_argument("--user", type=str, default=None, help="User ID (대화 이력 유지)")
    parser.add_argument("--interactive", action="store_true", help="대화 모드")

    args = parser.parse_args() # 명령행 인자 파싱

    config.setup_logging()     # 로깅 호출 - 로그 양식 설정(디폴트 INFO) -> DEBUG로 보고싶을 때 : config.setup_logging(logging.DEBUG)
    config.validate()          # 필수 설정 검증 + 폴더 생성(API 키 + 폴더 확인)


    # 1) 데이터 인덱싱 : 문서 로딩 + 전처리 + 청킹 → 청크 리스트 → 임베딩 → DB 저장
    if args.ingest: 
        n = ingest(reset=args.reset) # ingetst_all() -> retriever.add_chunks() 호출 -> DB 저장
        print(f"\n인덱싱 완료: {n}개 청크\n")

    # 단발 질문
    if args.query: 
        result = query(args.query, user_id=args.user) 
        print(f"\n[질문]: {args.query}")
        print(f"[User]: {result['user_id']}")
        if result["tool_calls"]:
            print("[도구 호출]:")
            for tc in result["tool_calls"]:     
                print(f"  - {tc['tool']}({tc['args']})")
        print(f"\n[답변]:\n{result['answer']}\n")
        print()


    # 대화 모드
    if args.interactive:
        if args.user:
            user_id = args.user
        else:
            user_input = input("User ID (엔터 시 자동 생성): ").strip()
            user_id = user_input if user_input else str(uuid.uuid4())

        print(f"\n=== 대화 모드 (User ID: {user_id}) ===")
        print("'exit' 또는 '종료' 입력 시 종료\n")

        while True:
            try:
                user_message = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n종료합니다.")
                break

            if user_message.lower() in ("exit", "quit", "종료"):
                print("대화 종료")
                break
            if not user_message:
                continue

            try:
                result = query(user_message, user_id=user_id)
                print(f"\nBot: {result['answer']}\n")
            except Exception as e:
                print(f"\n에러: {e}\n")

    if not any([args.ingest, args.query, args.interactive]):
        parser.print_help()


if __name__ == "__main__":
    main()
