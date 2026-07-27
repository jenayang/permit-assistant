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
from src.agent import extract_text, graph
from src.ingestion import ingest_all
from src.retriever import index_documents, count_documents, reset_collection
from src import parent_store

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

    logger.info("[User: %s] 답변 길이: %d, 도구 호출: %d회",
                user_id, len(answer), len(tool_calls))

    return {
        "answer": answer,
        "user_id": user_id,
        "tool_calls": tool_calls,
        "contexts": contexts,
        "permit_result": result.get("permit_result"),
        "food_result": result.get("food_result"),
        "fire_result": result.get("fire_result"),
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
