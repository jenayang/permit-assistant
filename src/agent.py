"""LangGraph 기반 ReAct 에이전트.

그래프 구조:
    START → agent → [도구 필요?]
                    ├─ 실제 도구 호출 있음 → tools → agent (순환)
                    ├─ case_facts 다 모였고 아직 미분류 → classify → agent (한 번 더,
                    │  분류 결과를 답변에 자연스럽게 반영하도록)
                    └─ 자유 텍스트 답변 → guard → [위반?]
                                                 ├─ 있음(재시도 1회 이내) → agent
                                                 ├─ 없음+분류 끝났고 미종합 → finalize → END
                                                 └─ 없음+그 외 → END

classify/finalize/guard는 모두 LLM이 아니라 그래프가 강제로 실행하는 결정론적 노드다:
- classify: 파이썬 규칙 함수(src.agents.permit.classify_case)로 허가/신고/기재변경을
  판정 - 법령상 객관적 기준으로 정해지는 값을 LLM 재량에 맡기지 않기 위함(LLM이
  도구 호출을 빼먹거나 기준을 잘못 계산하는 문제를 실제로 겪은 뒤 도입).
- finalize: classify가 확정한 permit_type/procedures + LLM이 record_permit_synthesis로
  채운 required_documents/related_agencies + 최종 답변 텍스트를 모아 PermitResult로
  조립. 판정(rule)과 종합(LLM)의 경계를 그래프 단계로 명확히 나눈다.
  required_documents/related_laws가 하나라도 채워지기 전까지는(=아직 owner_check
  같은 분기 질문 단계) 매 턴 다시 실행해서 최신 상태로 갱신하고, 한 번이라도
  채워지면(=실제 종합이 끝난 시점) 그 결과를 잠가서 이후 무관한 대화가 덮어쓰지
  못하게 한다 - 자세한 배경은 finalize_node docstring 참고.
- guard: 자유 텍스트로 끝나는 모든 답변을 정규식+상태로 검증한다(disclosed_stage
  상한 초과, 검색 없이 지어낸 [출처: ...] 인용). 프롬프트 지시만으로는 LLM이
  한 턴에 여러 단계를 몰아서 공개하거나 근거 없이 답하는 걸 못 막는다는 게
  2026-07-26 실사용 세션에서 재현돼서 도입 - 자세한 배경은 guard_node docstring
  참고.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from typing import Annotated

from langchain_cerebras import ChatCerebras
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src import config
from src.agents import TOOLS
from src.agents.food_safety import classify_food_business, food_business_message
from src.agents.permit import (
    PROCEDURE_TREE,
    PermitResult,
    build_permit_result,
    classify_case,
    procedure_stage_message,
)

logger = logging.getLogger(__name__)


# === 그래프 상태 ===(MessagesState + 대화 중 파악된 사용자 상황 사실들)
def merge_facts(existing: dict, update: dict) -> dict:
    """case_facts 채널의 reducer. update에서 None이 아닌 값만 덮어써서 누적한다
    (record_case_facts가 매 턴 일부 필드만 채워 보내도 이전 값이 안 지워지도록)."""
    return {**existing, **{k: v for k, v in update.items() if v is not None}}


class AgentState(MessagesState):
    case_facts: Annotated[dict, merge_facts]
    # record_permit_synthesis로 기록된 필수서류ᆞ협의기관. case_facts와 merge
    # 로직이 동일해서(None 아닌 키만 누적) merge_facts를 그대로 재사용한다.
    permit_synthesis: Annotated[dict, merge_facts]
    # finalize_node가 한 번만 채우는 최종 구조화 결과. 리듀서 없이 기본
    # 덮어쓰기로 충분(여러 노드가 동시에 쓰지 않음).
    permit_result: PermitResult | None
    # 식품위생법 판정용 - 건축(case_facts/permit_result)과 독립된 별도 도메인이라
    # 채널도 분리한다(카페 창업처럼 두 판정이 한 대화에서 동시에 필요할 수 있음).
    # food_result는 permit_result와 달리 다단계 종합(finalize)이 없어 classify
    # 노드가 판정과 동시에 바로 채운다 - required_documents 같은 후속 종합은
    # 이번 스코프 밖(Step 3b는 분류까지만).
    food_facts: Annotated[dict, merge_facts]
    food_result: str | None
    # 4단계(상황분석/서류/사전진단/기간) 안내 중 실제로 사용자에게 공개된
    # 최대 단계(0~4). 프롬프트 지시만으로는 LLM이 한 턴에 4단계를 전부
    # 쏟아내는 걸 못 막아서(2026-07-26 실사용 세션에서 재현, guard_node 참고)
    # 이 값으로 "이번 턴엔 몇 단계까지만" 상한을 그래프가 강제한다. 리듀서
    # 없이 기본 덮어쓰기(guard_node가 한 번에 하나씩만 갱신).
    disclosed_stage: int


# === 시스템 프롬프트 ===
SYSTEM_PROMPT = """당신은 서울시 건축 인허가 전문 어시스턴트입니다.
건축ᆞ용도변경 등 인허가 절차를 건축법ᆞ시행령ᆞ시행규칙ᆞ서울시 조례
기준으로 정확히 안내하고, 카페ᆞ음식점 등 식품접객업 창업 시 필요한
식품위생법상 영업신고 종류도 함께 판정해 안내합니다.

## 1. 정보 수집
질문에서 지역ᆞ시설 유형ᆞ행위 유형(신축ᆞ증축ᆞ개축ᆞ재축ᆞ이전ᆞ대수선ᆞ
용도변경ᆞ일반수선ᆞ가설건축물)ᆞ판정에 필요한 사실(규모ᆞ층수ᆞ용도지역ᆞ대수선
해당 여부ᆞ시설군 등, 상황에 맞는 것만)ᆞ소유/임차 여부를 파악해 그 턴에
record_case_facts로 기록하세요(알게 되는 대로 부분 호출해도 누적됨 - 사용자가
이미 말한 수치를 빠뜨리면 판정이 안 됩니다).

**한 메시지에서 뽑아낼 수 있는 사실은 그 턴에 전부(다른 도구 호출과 같은 턴에
함께) 기록하세요 - 미루면 판정이 그만큼 늦어지고, 그 사이 턴에 결론을 지어내는
실수로 이어집니다.** 특히 시설군(예: "카페"→7, 시설군 매핑표 참고)은 검색이나
외부 조회 없이 그 자리에서 바로 계산 가능하니 나중으로 미루지 마세요.

**허가/신고/기재변경 여부는 절대 사용자에게 묻지도, 당신이 계산하지도 마세요.**
법령상 객관적 기준(면적ᆞ층수ᆞ시설군)으로 시스템이 자동 판정하며, 결과는
다음 턴에 도구 응답으로 옵니다 - 그걸 참고해서 답변에 반영하세요. 정보가
부족하면 결론을 암시하지 말고 부족한 사실만 되물으세요. **판정 결과가 도구
응답으로 실제로 온 적이 없다면(=permit_type이 아직 확정 안 됨), [1단계]~[4단계]
절차ᆞ서류ᆞ사전진단 내용을 먼저 설명하지 마세요** - 아직 판정도 안 났는데
그 내용부터 말하면 근거 없는 추측이 됩니다(record_case_facts만 부분적으로
호출하고 판정 도구 응답 없이 4단계까지 답변해버리는 실수가 실제로
있었습니다 - 2026-07-24 확인). 특히 **85㎡ 기준은
증축ᆞ개축ᆞ재축 전용이며 신축에는 적용되지 않습니다** - 신축의 신고 대상
여부는 용도지역(관리ᆞ농림ᆞ자연환경보전지역)ᆞ연면적 200㎡ 미만ᆞ층수 3층
미만을 모두 봐야 합니다. **용도변경 판정에는 act_type="용도변경"ᆞ
current_facility_group(현재 건물의 등록 용도)ᆞdesired_facility_group(새로
하려는 업종) 세 개가 전부 필요합니다** - 셋 중 하나라도 빠지면 판정 자체가
영영 안 됩니다(실제로 겪은 문제: current_facility_group만 기록하고
desired_facility_group을 빠뜨려서 판정이 안 된 사례, act_type 자체를
안 적어서 판정이 안 된 사례 둘 다 있었음). lookup_building_ledger로
현재 용도(current_facility_group)를 조회했다면, **새로 하려는 업종의
desired_facility_group도 같은 턴에 바로 기록하세요** - 이건 시설군
매핑표에서 즉시 계산 가능하니(예: "카페"→7) 대장 조회 결과를 기다릴
필요가 없습니다. current_facility_group만 채우고 desired_facility_group을
나중으로 미루지 마세요. 하나만 아는 상태에서 "기재변경일 가능성이 높다"처럼
예상을 언급하지도 마세요.

**용도변경 판정 사유(시설군 이동 방향)는 재구성하지 말고 그대로 옮기세요.**
도구 응답에 "(사유: OO시설군(n)→XX시설군(m)는 ... 이동)"처럼 이미 계산ᆞ검증된
문장이 같이 옵니다 - 어순만 자연스럽게 다듬어 답변에 넣되, "번호가 작아지니
하위군" 같은 식으로 방향을 스스로 다시 판단해서 새로 문장을 만들지 마세요.
이미 맞는 문장이 있는데 왜 다시 판단하냐고 물을 수 있는데, 정확히 그 이유
때문입니다 - 시설군 번호는 작을수록 상위군이라 직관과 반대라서, 이 문장을
참고만 하고 나름대로 재해석하면 실제로 반복해서 방향을 뒤집어 말하는
사례가 있었습니다(허가↔신고 반대로 결론).

예외 - 용도지역: 일반인은 대부분 모르니 직접 묻지 말고, 주소를 알면 먼저
lookup_land_zone으로 자동 조회해서 성공 시 바로 record_case_facts로
기록하세요. 자동 조회가 실패했을 때만 사용자에게 직접 물어보세요.

카페ᆞ식당처럼 음식류를 조리ᆞ판매하는 업종이면, 식품위생법상 어떤 영업신고
대상인지 판단하는 데 필요한 사실(조리ᆞ판매 여부ᆞ완제품만 파는지ᆞ베이커리
위주인지ᆞ주류 판매 여부)도 파악되는 대로 record_food_facts로 기록하세요.
사무실ᆞ미용실처럼 식품위생법과 무관한 업종이 명백하면 serves_food=False만
기록해도 됩니다. **휴게음식점/일반음식점 같은 영업 종류 자체는 절대 묻거나
당신이 판단하지 마세요** - 식품위생법 시행령 제21조 기준으로 시스템이 자동
판정합니다. 대신 사용자가 실제로 답할 수 있는 사실만 물어보되, **그 질문이
어떤 결과를 가르는 기준인지 짧게 함께 알려주세요** (예: "주류도 함께
판매하시나요? 음주 허용 여부에 따라 휴게음식점/일반음식점 신고가 달라져서요").

## 2. 도구 선택
- 법률 용어 정의("OO이 뭐야") → search_by_term (핵심 용어만 추출)
- 절차ᆞ조건ᆞ서류 등 일반 질문 → search_regulations
- 용도지역을 모르는데 주소는 아는 경우 → lookup_land_zone
- 건폐율ㆍ용적률 질문 → lookup_building_ratio_limits (세부 용도지역을
  모르면 계산하지 말고 직접 질문)
- 용도변경ᆞ대수선ᆞ증축처럼 **기존 건물이 있는** 케이스에서 그 건물의 현재
  용도ᆞ연면적ᆞ층수를 사용자가 아직 말 안 해서 모르면 → lookup_building_ledger
  (주소, 번지 포함)로 채우세요. 신축(빈 땅)에는 대장이 없으니 쓰지 마세요.
  **사용자가 현재 용도를 이미 말해서 판정이 끝났어도, 주소를 알거나 받으면
  lookup_building_ledger로 건축물대장상 공식 용도를 조회해서 대조하세요** -
  일반인은 자기 건물의 공식 등록 용도를 모르거나 착각하는 경우가 흔합니다
  (예: 실제로는 "사무소"로 등록돼 있는데 본인은 "그냥 사무실 자리"로만
  인지). 조회 결과가 사용자가 말한 용도와 다르면:
  1) 불일치를 사용자에게 명확히 알리고, 실제 절차 판단은 공식 기록 기준을
     따라야 함을 설명하세요.
  2) 공식 기록에 맞는 시설군으로 record_case_facts를 다시 호출해
     current_facility_group을 정정하세요(재판정이 자동으로 다시 일어납니다).
  다만 **이 대조 자체가 1~2단계(상황 분석ᆞ필수 서류) 안내를 막는 필수
  조건은 아닙니다** - 주소를 모르면 사용자가 말한 정보로 먼저 1~2단계를
  안내하고, 정확성 확인 차원에서 "정확한 판정을 위해 주소를 확인해드릴까요?"
  처럼 병행 질문으로 자연스럽게 물어보세요.
- 여러 관점에서 검색이 필요하면 도구를 반복 사용하세요. 검색 결과에 법령
  계층(법/시행령/시행규칙/조례)이 섞여 다르게 말하면 법 > 시행령 > 시행규칙
  우선 원칙을 따르고, 조례는 상충이 아니라 지역 추가 규정으로 안내하세요.

## 3. 답변 작성
**(A) 판정 정보가 아직 부족함**: 부족한 사실을 되묻는 1~2문장으로 끝내세요.
`[1단계]` 같은 헤더나 절차ᆞ서류 설명은 이번 턴에 꺼내지 마세요.

**(B) 판정 완료(도구 응답으로 옴) 또는 판정이 필요 없는 질문**(정의 질문 등):
아래 4단계로 나눠 순서대로 안내하세요. 각 단계는 핵심만 간결하게 - 이전
단계/턴에서 말한 내용을 다시 설명하지 마세요. 각 단계 끝에 다음 단계를
계속 안내할지 짧게 묻고, 사용자가 동의하거나 관련 질문을 이어가면 다음
단계로 넘어가세요. "한 번에 다 알려줘" 요청 시에만 4단계를 모두 한 번에.
[2단계: 필수 서류]를 아직 검색 안 했다면 search_regulations로 근거를
확보하고 record_permit_synthesis로 기록하세요(required_documents,
related_agencies). [3단계: 사전 진단]도 마찬가지로, 안내한 항목들을
같은 도구의 pre_diagnosis_items에 답변과 동일한 문구로 기록하세요 -
프론트엔드 진행 표시가 이 두 목록의 존재 여부로 단계 완료를 판단하니,
텍스트로만 안내하고 기록을 빠뜨리면 안 됩니다.

- [1단계: 상황 분석+필요 절차] 한 줄 요약, 단계명+관할기관만 나열(설명 1줄 이내)
- [2단계: 필수 서류] 리스트만(설명 없이) - record_permit_synthesis(required_documents=[...])
- [3단계: 사전 진단] 주차대수ᆞ정화조 용량ᆞ소방시설ᆞ장애인 편의시설ᆞ위생
  요구사항 중 실제 해당하는 것만 1~2줄+근거와 함께 - record_permit_synthesis(pre_diagnosis_items=[...])
- [4단계: 예상 소요 기간] 한 줄로

## 4. 정확성 원칙
- 모든 답변에 [출처: 건축법 제OO조] / [출처: 서울시 건축조례 제OO조] 형식으로
  근거를 명시하세요. search_regulations/search_by_term으로 실제 검색하지
  않은 조항은 절대 지어내지 마세요.
- 확인되지 않는 정보는 "관련 규정에서 확인할 수 없습니다"라고 답하세요.

주의사항: 최종 판단은 관할 구청ᆞ건축사 상담을 권장하고, 최신 개정 여부는
국가법령정보센터(law.go.kr) 확인을 권장하세요. 개별 사안의 세부 판단은
전문가 상담이 필요합니다.
"""

# === LLM + 도구 바인딩 ===
llm = ChatGoogleGenerativeAI(
    model = config.GEMINI_MODEL,
    google_api_key = config.GEMINI_API_KEY,
    temperature = config.TEMPERATURE,
    max_output_tokens = config.MAX_OUTPUT_TOKENS,
    thinking_budget = config.THINKING_BUDGET,
)
llm_with_tools = llm.bind_tools(TOOLS)   # TOOL을 LLM에 바인딩. 제공함

# Cerebras는 실제로 폴백이 발생할 때만 생성 (CEREBRAS_API_KEY가 없으면 생성 시점에 에러남)
_cerebras_llm_with_tools = None

# 프로세스 내에서 Gemini 할당량 소진이 한 번 확인되면 이후 요청은 바로 Cerebras로 보냄
# (매번 Gemini를 재시도하며 429를 반복 유발하지 않기 위함, 재시작 시 초기화)
_gemini_quota_exhausted = False


def _is_quota_error(exc: Exception) -> bool:
    """Gemini 무료 티어 일일 할당량 초과(429 RESOURCE_EXHAUSTED) 여부 판별."""
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg


def _get_cerebras_llm_with_tools():
    global _cerebras_llm_with_tools
    if _cerebras_llm_with_tools is None:
        cerebras_llm = ChatCerebras(
            model = config.CEREBRAS_MODEL,
            api_key = config.CEREBRAS_API_KEY,
            temperature = config.TEMPERATURE,
            max_tokens = config.MAX_OUTPUT_TOKENS,
        )
        _cerebras_llm_with_tools = cerebras_llm.bind_tools(TOOLS)
    return _cerebras_llm_with_tools


# === 노드 정의 ===
def agent_node(state: AgentState) -> dict:
    """LLM을 호출해서 다음 액션 결정.

    - 도구가 필요하면 tool_calls를 반환
    - 답변 가능하면 최종 답변 반환
    - Gemini 무료 티어 할당량(하루 20회) 소진 시 Cerebras(config.CEREBRAS_MODEL)로 자동 전환
    - 응답에 record_case_facts 호출이 있으면 그 인자를 case_facts에 병합
    - 판정이 끝났으면 disclosed_stage 기준으로 "이번 턴엔 N단계까지만" 동적 지시를
      덧붙인다(연성 유도 - 실제 차단은 guard_node가 담당).
    """
    global _gemini_quota_exhausted
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if state.get("case_facts", {}).get("_classified"):
        disclosed = state.get("disclosed_stage", 0)
        if disclosed < 4:
            messages.append(SystemMessage(content=(
                f"[진행 상태] 지금까지 {disclosed}단계까지 공개했습니다. 이번 턴에는 "
                f"[{disclosed + 1}단계]까지만 안내하고, 그 이상 단계는 절대 먼저 꺼내지 "
                f"마세요 - 사용자가 이어서 요청하면 다음 턴에 공개하세요."
            )))
        else:
            messages.append(SystemMessage(content="[진행 상태] 4단계 안내가 모두 끝났습니다."))
    messages.extend(state["messages"])

    if not _gemini_quota_exhausted:
        try:
            response = llm_with_tools.invoke(messages)
            return _agent_result(response)
        except Exception as exc:
            if not _is_quota_error(exc):
                raise
            logger.warning(
                "Gemini 무료 티어 할당량 소진 감지. 이후 요청은 Cerebras(%s)로 전환합니다.",
                config.CEREBRAS_MODEL,
            )
            _gemini_quota_exhausted = True

    response = _get_cerebras_llm_with_tools().invoke(messages)
    return _agent_result(response)


def _agent_result(response: AIMessage) -> dict:
    """LLM 응답에서 record_case_facts/record_permit_synthesis/record_food_facts
    호출을 찾아 각각 case_facts/permit_synthesis/food_facts 갱신분으로 뽑아낸다.

    두 도구 자체는 여전히 ToolNode가 정상 실행해서(확인 문자열만 반환) 도구
    호출-응답 짝은 그대로 맞춰지고, 여기서는 그 인자를 그래프 상태에도
    반영하는 부수 작업만 한다. 새 record_* 도구를 추가할 때 여기 분기를
    같이 안 늘리면, 도구는 호출됐는데 상태엔 하나도 안 남는 채로 조용히
    무시된다(실제로 record_food_facts 추가 때 한 번 빠뜨렸다가 겪음 -
    food_facts가 항상 {}로 남아 classify가 영영 안 걸리는 버그였음).
    """
    facts_update: dict = {}
    synthesis_update: dict = {}
    food_facts_update: dict = {}
    for tc in getattr(response, "tool_calls", None) or []:
        if tc["name"] == "record_case_facts":
            facts_update.update(tc["args"])
        elif tc["name"] == "record_permit_synthesis":
            synthesis_update.update(tc["args"])
        elif tc["name"] == "record_food_facts":
            food_facts_update.update(tc["args"])
    result: dict = {"messages": [response]}
    if facts_update:
        result["case_facts"] = facts_update
    if synthesis_update:
        result["permit_synthesis"] = synthesis_update
    if food_facts_update:
        result["food_facts"] = food_facts_update
    return result


def classify_node(state: AgentState) -> dict:
    """case_facts/food_facts가 각각 충분히 모이면 해당 도메인을 결정론적으로
    분류하고, 그 결과를 (LLM이 부른 게 아니라 이 노드가 직접 구성한) tool_calls
    모양 메시지로 기록한다. pipeline.py의 tool_calls 추출 로직과 프론트의
    로드맵 렌더링 코드가 "tool: set_procedure_stage" 모양만 보고 반응하므로,
    별도 API/프론트 수정 없이 그대로 재사용된다.

    두 도메인(건축ᆞ식품위생)은 서로 독립적이라 한 턴에 한쪽만, 둘 다, 혹은
    (route_after_tools가 이미 걸러줘서) 아무것도 새로 분류되지 않을 수 있다 -
    해당하는 도메인만 골라 처리하고 메시지를 이어붙인다.
    """
    messages: list = []
    update: dict = {}

    facts = state.get("case_facts", {})
    if not facts.get("_classified"):
        result_type = classify_case(facts)
        if result_type is not None:
            root_node_id = PROCEDURE_TREE[result_type]["root"]
            call_id = f"classify_{uuid.uuid4().hex[:8]}"
            messages.append(AIMessage(
                content="",
                tool_calls=[{
                    "name": "set_procedure_stage",
                    "args": {"case_type": result_type, "node_id": root_node_id},
                    "id": call_id,
                }],
            ))
            messages.append(ToolMessage(
                content=procedure_stage_message(result_type, root_node_id, facts=facts),
                tool_call_id=call_id,
                name="set_procedure_stage",
            ))
            update["case_facts"] = {"_classified": True}
            logger.info("[classify] case_facts=%r → %s", facts, result_type)

    food_facts = state.get("food_facts", {})
    if not food_facts.get("_classified"):
        business_type = classify_food_business(food_facts)
        if business_type is not None:
            call_id = f"classify_food_{uuid.uuid4().hex[:8]}"
            messages.append(AIMessage(
                content="",
                tool_calls=[{
                    "name": "set_food_business_type",
                    "args": {"business_type": business_type},
                    "id": call_id,
                }],
            ))
            messages.append(ToolMessage(
                content=food_business_message(business_type),
                tool_call_id=call_id,
                name="set_food_business_type",
            ))
            update["food_facts"] = {"_classified": True}
            update["food_result"] = business_type
            logger.info("[classify_food] food_facts=%r → %s", food_facts, business_type)

    if messages:
        update["messages"] = messages
    return update


_CITATION_PATTERN = re.compile(r"\[출처:\s*([^\]]+)\]")
# 줄 시작(마크다운 헤딩 기호 허용)에 오는 [N단계]만 "실제 공개"로 센다.
# "다음 단계인 [2단계: 필수 서류] 안내를 계속할까요?"처럼 문장 중간에서 다음
# 단계를 이름만 언급하는 건 실제 공개가 아니므로 걸러야 한다(2026-07-26 guard
# 첫 실사용 검증에서 이 구분이 없어 오탐이 난 걸 확인하고 라인 앵커 추가).
_STAGE_MARKER_PATTERN = re.compile(r"^#{0,3}\s*\[([1-4])단계", re.MULTILINE)


def extract_text(content) -> str:
    """AIMessage.content에서 순수 텍스트만 뽑는다. 최신 Gemini는 content가
    단순 문자열이 아니라 [{"type": "text", "text": "..."}] 같은 블록 리스트로
    올 수 있어서(pipeline.py의 answer 추출과 동일한 처리 필요) 여기 한 곳에만
    두고 pipeline.py도 이 함수를 가져다 쓰게 해서 두 곳이 서로 다르게
    처리하다 어긋나는 걸 막는다(실제로 한 번 어긋나서 겪음).
    """
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return content or ""


def finalize_node(state: AgentState) -> dict:
    """PermitResult를 조립한다(LLM 재호출 없는 순수 결정론적 노드).
    permit_type/procedures는 build_permit_result()가 규칙 엔진으로 그대로
    채우고, required_documents/pre_diagnosis_items/related_laws/explanation은
    LLM이 record_permit_synthesis로 기록한 값 + 방금 낸 최종 답변에서 뽑는다.

    잠금(`_finalized`) 조건을 세 번 잘못 잡아본 뒤 정착한 버전이다:
    1) "분류 직후 딱 한 번만 실행 후 영구 잠금" - 분류 직후 첫 응답이 실제로는
       owner_check 같은 분기 질문이라 아직 아무것도 안 채워진 시점에 결과가
       굳어버리고, 이후 턴에서 search_regulations로 근거를 실제로 찾아도
       permit_result가 다시는 갱신되지 않았다.
    2) "분류된 이후 매 턴 무조건 재실행"(잠금 완전 제거) - 위 문제는 풀리지만,
       종합이 끝난 뒤 사용자가 "감사합니다" 같은 무관한 말을 하면 그 턴에도
       또 실행되어 explanation이 방금 만든 완성된 설명 대신 그 잡담 텍스트로
       덮어써지는 새 문제가 생겼다.
    3) "required_documents가 채워지면 잠금" - related_laws는 잠금 신호로 안
       쓰겠다고 docstring에는 적어놓고 실제 조건문은 안 고쳐서 한동안 여전히
       `or result.related_laws`가 남아있었다([출처: ...] 인용 하나만 있어도
       서류가 비어있는 채로 조기에 잠기는 버그) - 코드 리뷰 없이 docstring만
       고치면 실제로 이렇게 어긋난다는 걸 보여주는 사례라 남겨둔다.
    지금 버전: required_documents와 pre_diagnosis_items가 **둘 다** 채워지기
    전까지는(=아직 분기 질문이거나 서류ᆞ사전진단 중 하나만 끝난 미완성 단계)
    매 턴 다시 실행해서 최신 상태로 갱신하고, 둘 다 채워지면(=SYSTEM_PROMPT가
    유도하는 2ᆞ3단계가 서로 다른 턴에 걸쳐 나올 수 있어서 하나만으로는
    부족함) 그 결과를 최종 스냅샷으로 잠가서 이후의 무관한 대화가 덮어쓰지
    못하게 한다. related_laws는 여전히 잠금 신호로 안 쓴다(2026-07-24 실측
    확인, 위 3번 참고).
    """
    facts = state.get("case_facts", {})
    result = build_permit_result(facts)
    if result is None:
        return {}

    synthesis = state.get("permit_synthesis", {})
    last_content = extract_text(state["messages"][-1].content)

    result.required_documents = synthesis.get("required_documents", [])
    result.pre_diagnosis_items = synthesis.get("pre_diagnosis_items", [])
    result.related_laws = sorted(set(_CITATION_PATTERN.findall(last_content)))
    result.explanation = last_content

    logger.info(
        "[finalize] permit_type=%s, required_documents=%d, pre_diagnosis_items=%d, related_laws=%d",
        result.permit_type, len(result.required_documents),
        len(result.pre_diagnosis_items), len(result.related_laws),
    )

    update: dict = {"permit_result": result}
    if result.required_documents and result.pre_diagnosis_items:
        update["case_facts"] = {"_finalized": True}
    return update


def _max_allowed_stage(state: AgentState) -> int:
    """이번 턴 답변에 등장해도 되는 최대 [N단계] 번호. 판정 전에는 0(전부 금지),
    판정 후에는 지금까지 공개된 단계(disclosed_stage) + 1 - 한 턴에 최대 한
    단계만 새로 공개하도록 강제한다."""
    if not state.get("case_facts", {}).get("_classified"):
        return 0
    return state.get("disclosed_stage", 0) + 1


def _has_grounding_search(messages) -> bool:
    """대화 전체에 실제 법령 검색(search_regulations/search_by_term) 호출이
    있었는지. 대화 전체를 스캔하므로 훨씬 이전 턴의 검색으로 나중 턴의 무관한
    인용까지 통과될 수 있다는 한계는 있지만, "검색을 아예 한 번도 안 하고
    조항을 지어내는" 가장 심각한 사례는 확실히 잡는다(1차 구현, 2026-07-26)."""
    return any(
        isinstance(m, ToolMessage) and m.name in ("search_regulations", "search_by_term")
        for m in messages
    )


def _guard_retry_count(messages) -> int:
    """이번 사용자 턴(마지막 HumanMessage 이후) 안에서 guard가 이미 몇 번
    재시도를 유도했는지. 무한 루프 방지용 - 1회를 넘으면 그냥 통과시킨다."""
    count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage) and m.name == "_answer_guard":
            count += 1
    return count


def guard_node(state: AgentState) -> dict:
    """agent가 자유 텍스트로 답을 끝내려 할 때, 근거ᆞ판정 없이 앞서나간 답변을
    한 번 걸러낸다. LLM 판단이 아니라 정규식+상태로 결정론적으로 감지한다
    (classify_case와 같은 원칙 - 프롬프트 지시만으로는 못 막는다는 게
    2026-07-26 실사용 세션에서 재현됨).

    막는 위반 두 가지:
    1. 허용된 단계 수([_max_allowed_stage])를 넘겨 [N단계]를 안내 - 판정 전
       절차 안내를 아예 시도한 경우(허용치 0)와, 판정 후 한 턴에 여러 단계를
       몰아서 공개한 경우(사용자가 "1단계만" 이라고 명시해도 무시하고 4단계를
       다 준 사례 포함) 둘 다 이 하나의 규칙으로 잡는다.
    2. search_regulations/search_by_term을 한 번도 호출하지 않았는데
       [출처: ...] 인용이 있는 경우 - 근거 없이 조항을 지어낸 것.

    같은 사용자 턴 안에서 최대 1회만 재시도를 유도한다(무한 루프 방지) - 그
    이상 반복되면 프롬프트만으로는 못 막는 한계로 보고 그냥 통과시킨다.
    위반이 없거나 재시도가 소진되면, 실제로 공개된 최대 단계로 disclosed_stage를
    갱신해서 다음 턴의 허용치 계산에 반영한다.
    """
    messages = state["messages"]
    last_text = extract_text(messages[-1].content)

    allowed = _max_allowed_stage(state)
    mentioned = [int(n) for n in _STAGE_MARKER_PATTERN.findall(last_text)]
    max_mentioned = max(mentioned, default=0)

    violations = []
    if max_mentioned > allowed:
        violations.append(
            f"이번 턴엔 [{allowed}단계]까지만 안내할 수 있는데 [{max_mentioned}단계]까지 "
            f"안내했습니다. 허용된 단계까지만 남기고 나머지는 삭제한 뒤, 다음에 계속 "
            f"안내해도 될지 사용자에게 짧게 물어보며 마무리하세요."
        )
    if _CITATION_PATTERN.search(last_text) and not _has_grounding_search(messages):
        violations.append(
            "search_regulations/search_by_term을 한 번도 호출하지 않았는데 [출처: ...] "
            "인용이 포함되어 있습니다. 검색 없이 조항을 지어내지 말고, 먼저 검색 도구를 "
            "호출해 실제 근거를 확보한 뒤 답변하세요."
        )

    if violations and _guard_retry_count(messages) < 1:
        logger.info("[guard] 위반 감지, 재시도 유도: %s", violations)
        call_id = f"guard_{uuid.uuid4().hex[:8]}"
        return {
            "messages": [
                AIMessage(content="", tool_calls=[{"name": "_answer_guard", "args": {}, "id": call_id}]),
                ToolMessage(
                    content="답변 재검토 필요:\n- " + "\n- ".join(violations),
                    tool_call_id=call_id,
                    name="_answer_guard",
                ),
            ]
        }

    if violations:
        logger.warning("[guard] 위반이 재시도 후에도 지속됨 - 통과: %s", violations)

    update: dict = {}
    if max_mentioned > state.get("disclosed_stage", 0):
        update["disclosed_stage"] = max_mentioned
    return update


def route_after_tools(state: AgentState) -> str:
    """tools 실행 직후: case_facts 또는 food_facts가 방금 완성됐으면(각각
    record_case_facts/record_food_facts로 채워짐) agent로 돌아가기 전에
    classify부터 강제한다.

    예전엔 이 체크를 route_after_agent(agent 응답 이후)에서만 했는데, 그러면
    "case_facts는 이미 다 모였지만 LLM이 그 사실을 모른 채 도구 호출 없이
    먼저 답변을 시도 -> 그래프가 그 답변을 무시하고 classify로 강제 전환 ->
    다음 턴 LLM이 '이미 답했나?' 헷갈려하며 부실한 후속 답변을 내는" 문제가
    실제로 발생했다. tools 직후로 당기면 LLM이 답변을 시도하기 전에 분류가
    먼저 끝나서 이 문제가 구조적으로 사라진다. food_facts도 동일한 이유로
    같은 시점에 체크한다(도메인이 늘어도 이 타이밍 원칙은 그대로 적용).
    """
    facts = state.get("case_facts", {})
    food_facts = state.get("food_facts", {})
    needs_classify = not facts.get("_classified") and classify_case(facts) is not None
    needs_food_classify = (
        not food_facts.get("_classified") and classify_food_business(food_facts) is not None
    )
    if needs_classify or needs_food_classify:
        logger.info(
            "[route_after_tools] case_facts=%r food_facts=%r → classify", facts, food_facts
        )
        return "classify"
    logger.info(
        "[route_after_tools] case_facts=%r food_facts=%r → agent (분류 조건 미충족 또는 이미 분류됨)",
        facts, food_facts,
    )
    return "agent"


def route_after_agent(state: AgentState) -> str:
    """agent 다음 어디로 갈지: 실제 도구 호출 > 자유 텍스트 답변은 무조건 guard.

    분류(classify)는 이제 route_after_tools에서 처리되므로, agent가 응답하는
    시점엔 이미 분류가 끝나 있는 게 정상이라 여기선 다시 체크하지 않는다.
    자유 텍스트로 답을 끝내려는 시도는 finalize/END로 바로 보내지 않고 항상
    guard를 거쳐 단계 상한ᆞ인용 근거를 먼저 검증한다.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        logger.info("[route_after_agent] tool_calls=%r → tools", [tc["name"] for tc in last.tool_calls])
        return "tools"

    logger.info("[route_after_agent] 자유 텍스트 답변 → guard")
    return "guard"


def route_after_guard(state: AgentState) -> str:
    """guard 다음 어디로 갈지: 재시도 유도 메시지가 방금 추가됐으면 agent로
    돌아가고, 아니면 (분류 끝났고 아직 미종합이면) finalize, 그 외엔 종료."""
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and last.name == "_answer_guard":
        logger.info("[route_after_guard] 재시도 유도 → agent")
        return "agent"

    facts = state.get("case_facts", {})
    if facts.get("_classified") and not facts.get("_finalized"):
        logger.info("[route_after_guard] case_facts=%r → finalize", facts)
        return "finalize"

    logger.info("[route_after_guard] case_facts=%r → END", facts)
    return END


# === ToolNode 생성 ===
tool_node = ToolNode(tools=TOOLS)   # tool 콜 받아서 실제 함수 실행


# === Graph 구성 ===
def build_graph():
    """ReAct 에이전트 그래프 구성.

    - checkpointer: 대화 이력 (user_id 기준)
    - store: 사용자 정보 저장소 (user_id 기준?, long-term)
    """
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("classify", classify_node)
    builder.add_node("guard", guard_node)
    builder.add_node("finalize", finalize_node)

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
    checkpointer = SqliteSaver(conn)  # short-term memory (영구 저장)
    checkpointer.setup()
    store = InMemoryStore()         # long-term memory (미사용 - 그대로 둠)
    return builder.compile(checkpointer=checkpointer, store=store)


# === 컴파일된 그래프 (모듈 로드 시 1회) ===
graph = build_graph()

# === Store 접근 헬퍼 (외부에서 저장/조회 가능하도록) ===
def get_store():
    """그래프에 연결된 store 반환.

    사용 예:
        from src.agent import get_store
        store = get_store()
        store.put(("user", "jena"), "language", {"value": "ko"})
        result = store.get(("user", "jena"), "language")
    """
    return graph.store
