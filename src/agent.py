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

import json
import logging
from dataclasses import dataclass
from typing import Annotated

from langchain_cerebras import ChatCerebras
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from src import config
from src.agents import TOOLS
from src.agents.permit import PermitResult, build_permit_result

logger = logging.getLogger(__name__)

# disclosed_stage(dict[domain, stage])에서 지금 유일하게 4단계(상황분석/서류/
# 사전진단/기간) 구조를 쓰는 도메인의 키. permit만 이 패턴을 쓴다 - food/fire/
# signage 등 나머지는 classify 결과 자체가 완결된 답이라 단계 구조가 없다
# (2026-07-27 논의: 두 번째로 이 패턴이 필요한 도메인이 실제로 생기면 그때
# DOMAIN_CONFIGS에 supports_stage 같은 필드로 일반화 - 지금은 이르다).
_STAGE_DOMAIN = "case_facts"


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
    # 소방시설 판정용 - 위 두 도메인과 마찬가지로 독립 채널. fire_result는
    # food_result(단일 문자열)와 달리 "필요한 소방시설 목록"이라 list다 - 규모가
    # 커지면 여러 시설이 동시에 필요할 수 있고, 빈 리스트([])도 "설치 대상 없음"
    # 이라는 유효한 판정이라 None(미분류)과 구분해야 한다.
    fire_facts: Annotated[dict, merge_facts]
    fire_result: list[str] | None
    # 간판ᆞ옥외광고물 허가/신고 판정용 - 위 도메인들과 마찬가지로 독립 채널.
    # food_result처럼 단일 문자열(허가/신고/허가ᆞ신고 불필요) - permit처럼
    # 별도 종합(finalize) 단계 없이 classify 시점에 바로 확정된다.
    signage_facts: Annotated[dict, merge_facts]
    signage_result: str | None
    # 4단계(상황분석/서류/사전진단/기간, 로드맵 Step 1의 하위 단계라 [Step 1-N]
    # 으로 표시 - Step 0~4 전체 번호와 겹치면 혼동된다는 피드백으로 2026-07-27
    # 개명) 안내 중 실제로 사용자에게 공개된 최대 단계(0~4) - 도메인별로
    # 분리된 dict다(키: DOMAIN_CONFIGS의 facts_key, 지금은 "case_facts"=permit
    # 만 실제로 씀). 처음엔 전역 스칼라 하나였는데, signage처럼 이 단계 구조를
    # 안 쓰는 도메인의 답변에 LLM이 [Step 1-N] 마커를 잘못 갖다 써도 그게
    # permit의 카운터를 오염시켜서
    # (예: 간판 얘기만 했는데 disclosed_stage가 올라가 나중에 진짜 건축
    # 얘기를 시작하면 이미 일부 단계가 끝난 것으로 오인) 도메인별로 분리했다
    # (2026-07-27 실측 확인). 프롬프트 지시만으로는 LLM이 한 턴에 4단계를
    # 전부 쏟아내는 걸 못 막아서(2026-07-26 실사용 세션에서 재현, guard_node
    # 참고) 이 값으로 "이번 턴엔 몇 단계까지만" 상한을 그래프가 강제한다.
    # 2026-07-28: "construction_guide_shown" 키를 추가해 같은 dict를 Step1→2
    # 전환 신호로도 재사용한다(bool - get_construction_guide가 대화 중 한 번이라도
    # 호출됐는지, guard_node가 갱신). task_progress의 "construction"(사용자가
    # 실제로 착공했다는 자기보고)과는 의미가 다르니 혼동하지 말 것 - 여긴
    # "안내를 이미 보여줬는지"만 본다. 2026-08-04: Step2도 [Step 2-N]으로
    # 세분화하며 "construction_stage"(int 0~3, 텍스트 공개 카운터)를 별도 키로
    # 추가했다 - case_facts 카운터와 같은 패턴. 자세한 이유는
    # permit_phase_directive 참고.
    # 리듀서 없이 기본 덮어쓰기(guard_node가 한 번에 하나씩만 갱신).
    disclosed_stage: dict[str, int]
    # 사용자가 실제로 "완료했다"고 말한 항목들(착공신고ᆞ사업자등록ᆞ영업개시 등).
    # permit_result/food_result 등(규칙 엔진이 계산한 "뭘 해야 하는지" 판정)과는
    # 성격이 다르다 - 이건 실물 세계에서 벌어지는 일이라 AI가 스스로 판단하거나
    # 다른 도구로 확인할 방법이 없고, 오직 사용자가 명시적으로 완료를 밝힌
    # 것만 기록한다("안내를 보여줬다"≠"완료했다" - 2026-07-27 논의). merge_facts
    # 재사용 - None 아닌 값만 누적, 한 번 True가 되면 계속 유지.
    task_progress: Annotated[dict, merge_facts]


# === 시스템 프롬프트(src/prompts.py로 분리, Phase 1) ===
from src.prompts import build_system_prompt  # noqa: E402
from src.message_utils import extract_text  # noqa: E402

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


class ContextOverflowError(RuntimeError):
    """컨텍스트 한도 초과로 LLM 호출이 실패한 경우.

    프로바이더 원본 에러를 그대로 500으로 흘리면 사용자에겐 의미 없는 영문
    스택이 노출되고, 무엇을 해야 할지도 알 수 없다. 이 타입으로 감싸서
    api.py가 "새 대화를 시작하라"는 행동 가능한 안내로 바꾼다.
    """


_CONTEXT_ERROR_MARKERS = (
    "context length", "context_length", "maximum context", "too many tokens",
    "token limit", "input is too long", "request too large", "reduce the length",
)


def _is_context_overflow_error(exc: Exception) -> bool:
    """컨텍스트 한도 초과 에러인지 판별. 프로바이더마다 문구가 달라 부분
    문자열로 넓게 잡는다 - 잘못 잡아도 결과는 "더 친절한 안내"라 안전한 방향."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _CONTEXT_ERROR_MARKERS)


# 도구 스키마(설명+args)는 매 호출 전송되지만 messages에는 없어서 따로 더한다.
# 모듈 로드 시 1회만 계산.
_TOOL_SCHEMA_CHARS = sum(
    len(t.description or "") + len(json.dumps(t.args, ensure_ascii=False)) for t in TOOLS
)


def estimate_tokens(messages) -> int:
    """LLM에 나가는 컨텍스트 크기 추정치(토큰).

    한국어 법령ᆞ대화 텍스트를 실측하니 cl100k 기준 1.02 문자/토큰으로 꽤
    일정해서, 문자 수를 그대로 토큰 수로 본다(약 2% 과대추정 - 가드 입장에선
    안전한 방향). tiktoken을 안 쓰는 이유는 두 가지다: 직접 선언한 의존성이
    아니라 전이 의존성이고, cl100k는 Gemini/Gemma의 토크나이저가 아니라
    어차피 근사치라 정밀도를 위해 의존성을 늘릴 이유가 없다.
    """
    total = _TOOL_SCHEMA_CHARS
    for m in messages:
        total += len(extract_text(getattr(m, "content", "")))
        for tc in getattr(m, "tool_calls", None) or []:
            total += len(str(tc.get("args", "")))
    return total


def _count_message_chars(messages) -> int:
    """trim_messages에 넘길 토큰 카운터(메시지 분량만 - 도구 스키마 제외).

    estimate_tokens와 달리 고정 오버헤드를 안 더한다 - trim_messages는 "메시지
    목록이 예산에 맞는지"만 보므로, 고정분은 호출부에서 예산에서 미리 빼둔다.
    """
    total = 0
    for m in messages:
        total += len(extract_text(getattr(m, "content", "")))
        for tc in getattr(m, "tool_calls", None) or []:
            total += len(str(tc.get("args", "")))
    return total


def _fit_context(prefix: list, history: list, using_fallback: bool) -> list:
    """한도에 가까우면 오래된 대화를 잘라낸 사본을 만든다. 평소엔 원본 그대로.

    설계 선택 세 가지:
    1. **예산 기준 트리거**(항상 자르지 않음) - 실제 대화는 한도의 2~3%라 평소엔
       이 코드가 아예 안 켜진다. 예전에 "항상 최근 4턴만" 방식으로 넣었다가
       효과를 증명 못 해 롤백한 적이 있는데(4-9), 그건 짧은 대화의 멀쩡한 맥락까지
       매번 버리는 방식이었다. 여기서는 한도 근접이라는 명확한 이유가 있을 때만
       동작하므로 위험 프로필이 다르다.
    2. **LLM에 보내는 사본만 자르고 state["messages"]는 보존** - guard_node가
       메시지 이력을 스캔해서 검색 여부(_has_grounding_search)ᆞ공사 안내 호출
       여부(_construction_guide_shown)를 판정하기 때문이다. 원본을 지우면 "검색
       안 했다"고 오판해 없는 위반을 만든다. 체크포인터에도 전체 이력이 남는다.
    3. **직접 구현 대신 langchain의 trim_messages 사용** - 유효한 이력 형태
       (HumanMessage로 시작)와 tool_call/ToolMessage 짝 맞추기를 알아서 처리한다.
       짝이 깨지면 프로바이더가 400을 뱉는데, 그걸 직접 관리하면 버그가 나기 쉽다.

    판정 결과ᆞ사실은 메시지가 아니라 별도 상태 채널(case_facts/permit_result 등)에
    있고 _roadmap_status_summary가 매 턴 현황을 다시 주입하므로, 오래된 메시지를
    빼도 "지금까지 뭐가 확정됐는지"는 그대로 전달된다 - 이 구조 덕분에 절삭이
    일반 챗봇보다 안전하다.
    """
    limit = config.CEREBRAS_CONTEXT_LIMIT if using_fallback else config.GEMINI_CONTEXT_LIMIT
    threshold = limit * config.CONTEXT_TRIM_RATIO
    if estimate_tokens(prefix + history) <= threshold:
        return history

    # 예산 = 임계치 - (도구 스키마 + 매 턴 새로 만드는 시스템 메시지 + 출력 여유)
    budget = int(
        threshold - _TOOL_SCHEMA_CHARS - _count_message_chars(prefix) - config.MAX_OUTPUT_TOKENS
    )
    if budget <= 0:
        logger.warning("[CONTEXT] 고정 오버헤드만으로 예산 초과 - 절삭 생략")
        return history

    trimmed = trim_messages(
        history,
        max_tokens=budget,
        token_counter=_count_message_chars,
        strategy="last",       # 최근 대화를 남기고 오래된 것부터 버린다
        start_on="human",      # 유효한 이력 형태 유지
        include_system=False,  # 시스템 메시지는 prefix로 따로 붙는다
        allow_partial=False,   # 메시지를 반 토막 내지 않는다
    )
    logger.warning(
        "[CONTEXT] 한도 근접으로 대화 이력 절삭: %d개 → %d개 (예산 %d) - "
        "판정 결과는 상태에 보존되며 원본 이력도 그대로 남습니다",
        len(history), len(trimmed), budget,
    )
    return trimmed


def _log_context_usage(messages, using_fallback: bool) -> None:
    """이번 호출의 컨텍스트 사용량을 남긴다(차단하지 않음).

    grep 하기 쉬운 고정 포맷:
      [CONTEXT] model=cerebras est=95,120 limit=131,072 pct=73%
    실제로 얼마나 자주 위험 구간에 가는지 데이터가 쌓여야 트리밍ᆞ요약 같은
    큰 설계를 도입할지 실측으로 판단할 수 있다 - 2026-08-02 시점 추정으로는
    로드맵 완주 대화가 한도의 70%라 아직 근거가 약하다고 보고 도입을 미뤘다.
    """
    est = estimate_tokens(messages)
    limit = config.CEREBRAS_CONTEXT_LIMIT if using_fallback else config.GEMINI_CONTEXT_LIMIT
    ratio = est / limit
    line = "[CONTEXT] model=%s est=%d limit=%d pct=%d%%"
    args = ("cerebras" if using_fallback else "gemini", est, limit, round(ratio * 100))
    if ratio >= config.CONTEXT_WARN_RATIO:
        logger.warning(line + " - 한도 근접", *args)
    else:
        logger.info(line, *args)


def _get_cerebras_llm_with_tools(tool_choice: str | None = None):
    """tool_choice가 없으면(평소 경로) 모듈 전역 싱글턴을 재사용한다. 특정
    도구를 강제할 때(tool_choice 지정)는 그때만 쓰는 상황이라 캐싱하지 않고
    매번 새로 바인딩한다 - 강제 바인딩을 캐싱해버리면 이후 평소 호출까지
    계속 그 도구만 강제되는 사고로 이어진다."""
    global _cerebras_llm_with_tools
    if tool_choice:
        cerebras_llm = ChatCerebras(
            model=config.CEREBRAS_MODEL,
            api_key=config.CEREBRAS_API_KEY,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )
        return cerebras_llm.bind_tools(TOOLS, tool_choice=tool_choice)
    if _cerebras_llm_with_tools is None:
        cerebras_llm = ChatCerebras(
            model = config.CEREBRAS_MODEL,
            api_key = config.CEREBRAS_API_KEY,
            temperature = config.TEMPERATURE,
            max_tokens = config.MAX_OUTPUT_TOKENS,
        )
        _cerebras_llm_with_tools = cerebras_llm.bind_tools(TOOLS)
    return _cerebras_llm_with_tools


# === 로드맵 단계 추적(src/roadmap.py) + 답변 신뢰성 검증(src/guard.py) 분리(Phase 4) ===
from src.roadmap import (  # noqa: E402
    _roadmap_status_summary,
    _should_force_construction_guide,
    domain_timing_directive,
    permit_phase_directive,
)
from src.guard import _CITATION_PATTERN  # noqa: E402
from src.agents.roadmap_progress import cascade_construction_progress  # noqa: E402


# 정의를 묻는 질문("건폐율이 뭐야")은 기록할 사실이 없어 강제 대상에서 뺀다.
# 빼는 방향은 안전하다 - 강제를 안 하면 그냥 지금까지의 동작(모델 재량)으로
# 돌아갈 뿐이다.
_DEFINITION_QUESTION_MARKERS = ("뭐야", "뭔가요", "무엇", "정의", "뜻이", "차이가")


@dataclass(frozen=True)
class _ForcedToolDecision:
    """이번 턴에 tool_choice로 무엇을 왜 강제했는지. 로그로 남겨서 나중에
    "강제가 실제로 얼마나 자주 필요한지 / 강제 후 무엇이 추출됐는지"를 분석할
    수 있게 한다 - 이 프로젝트는 실측으로 방향을 정해왔는데(4-9 트리밍 반증,
    4-12 원인 규명), 정작 강제 자체에 대한 데이터는 없었다.

    로그 포맷(grep 하기 쉽게 고정):
      [FORCED_TOOL] domain=food tool=record_food_facts reason=no_facts trigger='카페'
      [FORCED_TOOL_RESULT] tool=record_food_facts called=True args={'serves_food': True}
    """
    tool: str
    domain: str
    reason: str
    trigger: str


def _log_forced_result(decision: _ForcedToolDecision | None, response: AIMessage) -> None:
    """강제한 도구가 실제로 불렸는지 + 어떤 값이 추출됐는지 남긴다.

    called=False가 찍히면 tool_choice 강제조차 안 먹혔다는 뜻이라 즉시 조사 대상이고,
    args가 비어 있으면 "호출은 됐지만 추출은 실패"라 다음 과제(추출 품질)의 입력이
    된다 - 호출률과 추출 품질을 따로 볼 수 있게 일부러 두 값을 같이 찍는다.
    """
    if decision is None:
        return
    args = next(
        (tc["args"] for tc in (getattr(response, "tool_calls", None) or [])
         if tc["name"] == decision.tool),
        None,
    )
    logger.info(
        "[FORCED_TOOL_RESULT] tool=%s called=%s args=%r",
        decision.tool, args is not None, args if args is not None else {},
    )


def _forced_record_tool(state: AgentState) -> _ForcedToolDecision | None:
    """유저가 이번 턴에 어떤 도메인을 꺼냈는데 그 도메인 사실이 아직 하나도
    기록되지 않았으면, 그 도메인의 record_* 도구를 tool_choice로 강제한다.

    2026-08-02 실측으로 원인을 규명한 뒤 도입했다. LLM 1회 호출로 격리해서
    측정한 결과(시행 3회, 전부 동일 = 결정론적):
      - 운영과 동일한 15개 도구:        record_* 호출 0/3
      - 관련 도구 3개만 노출:            signage 3/3, food 0/3
      - tool_choice 강제:                둘 다 3/3
    즉 ① 모델이 도구를 "필수"가 아니라 "권장"으로 다루고 ② 도구 수가 많을수록
    선택이 흐려진다. 프롬프트로 "반드시 기록하라"고 아무리 적어도(실제로 여러 번
    강화했다) 안 고쳐지던 이유다. 반면 API 레벨 강제는 100% 재현됐다 -
    get_construction_guide 한 지점에만 좁게 쓰던 방식(_should_force_construction_guide)을
    모든 도메인으로 일반화한 것.

    강제 범위를 좁게 유지하는 세 조건:
    1. facts가 완전히 비어 있을 때만 - 도메인당 대화 전체에서 사실상 첫 기록
       한 번뿐이라 강제가 무한히 반복되지 않는다.
    2. 이번 턴에 그 도구가 아직 안 불렸을 때만 - 강제했는데 모델이 전부 None으로
       채워 보내면 facts가 여전히 비어 다음 호출에서 또 강제되는 무한 루프가
       생긴다. 턴 안에서 1회로 제한해 끊는다.
    3. 정의를 묻는 질문은 제외 - 기록할 사실 자체가 없다.
    """
    messages = state.get("messages") or []
    if not messages or not isinstance(messages[-1], HumanMessage):
        return None
    text = extract_text(messages[-1].content)
    if any(marker in text for marker in _DEFINITION_QUESTION_MARKERS):
        return None

    called_this_turn = set()
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage):
            called_this_turn.update(tc["name"] for tc in (m.tool_calls or []))

    for config in DOMAIN_CONFIGS:
        if state.get(config.facts_key):
            continue
        if config.record_tool in called_this_turn:
            continue
        trigger = next((kw for kw in config.keywords if kw in text), None)
        if trigger is not None:
            return _ForcedToolDecision(
                tool=config.record_tool,
                domain=config.domain,
                reason="no_facts",
                trigger=trigger,
            )
    return None


# === 노드 정의 ===
def agent_node(state: AgentState) -> dict:
    """LLM을 호출해서 다음 액션 결정.

    - 도구가 필요하면 tool_calls를 반환
    - 답변 가능하면 최종 답변 반환
    - Gemini 무료 티어 할당량(하루 20회) 소진 시 Cerebras(config.CEREBRAS_MODEL)로 자동 전환
    - 응답에 record_case_facts 호출이 있으면 그 인자를 case_facts에 병합
    - build_system_prompt로 이번 턴에 살아있는 도메인의 프롬프트 블록만
      조립해서 전달한다(파일 상단 블록 주석 참고).
    - permit_phase_directive로 건축 인허가 트랙 순차 구간(Step 1-N 상한ᆞ
      Step 1→2 전환)의 동적 지시를 덧붙인다(연성 유도 - 실제 차단은
      guard_node가 담당).
    - 매 턴 _roadmap_status_summary로 건축 인허가 트랙(순차)ᆞ창업 준비 트랙
      (병렬 가능)의 진행 상태를 함께 전달한다.
    - _should_force_construction_guide가 True면 tool_choice를 강제해서
      get_construction_guide 호출을 API 레벨에서 보장한다(2026-07-31 -
      프롬프트 지시만으로는 못 막는다는 게 반복 확인되어, guard_node의
      사후 검증에 이어 이번엔 애초에 스킵 자체가 불가능하게 만드는 접근).

    2026-07-30: 오래된 턴의 tool_call/ToolMessage를 걷어내는 컨텍스트
    트리밍(_trim_tool_noise)을 먼저 시도했다가 되돌렸다 - 실제 A/B 실험에서
    트리밍 유무와 무관하게 결과가 같아서(재현 자체가 안 됨) 효과를 증명하지
    못했다. 상세는 docs/project_report.md 4-9 참고.
    """
    global _gemini_quota_exhausted
    # prefix는 매 턴 상태에서 새로 만드는 시스템 메시지라 절삭 대상이 아니다
    # (여기에 로드맵 현황이 들어 있어서, 오래된 대화를 잘라도 "지금까지 뭐가
    # 확정됐는지"는 그대로 전달된다).
    prefix = [
        SystemMessage(content=build_system_prompt(state)),
        SystemMessage(content=_roadmap_status_summary(state)),
    ]
    directive = permit_phase_directive(state)
    if directive:
        prefix.append(SystemMessage(content=directive))
    timing_directive = domain_timing_directive(state)
    if timing_directive:
        prefix.append(SystemMessage(content=timing_directive))
    messages = prefix + _fit_context(prefix, list(state["messages"]), _gemini_quota_exhausted)

    # 공사 안내 강제가 우선 - Step1→2 전환은 그 턴에 반드시 짚어야 하는 지점이라,
    # 같은 턴에 기록 강제와 겹치면 전환 쪽을 먼저 처리하고 기록은 다음 턴에 맡긴다.
    if _should_force_construction_guide(state):
        decision = _ForcedToolDecision(
            tool="get_construction_guide", domain="construction",
            reason="step1_done_guide_unshown", trigger="(직전 AI가 Step 2 제안)",
        )
    else:
        decision = _forced_record_tool(state)
    force_tool = decision.tool if decision else None
    if decision:
        logger.info(
            "[FORCED_TOOL] domain=%s tool=%s reason=%s trigger=%r",
            decision.domain, decision.tool, decision.reason, decision.trigger,
        )

    if not _gemini_quota_exhausted:
        _log_context_usage(messages, using_fallback=False)
        try:
            bound = llm.bind_tools(TOOLS, tool_choice=force_tool) if force_tool else llm_with_tools
            response = bound.invoke(messages)
            _log_forced_result(decision, response)
            return _agent_result(response)
        except Exception as exc:
            if _is_context_overflow_error(exc):
                raise ContextOverflowError(str(exc)) from exc
            if not _is_quota_error(exc):
                raise
            logger.warning(
                "Gemini 무료 티어 할당량 소진 감지. 이후 요청은 Cerebras(%s)로 전환합니다.",
                config.CEREBRAS_MODEL,
            )
            _gemini_quota_exhausted = True

    _log_context_usage(messages, using_fallback=True)
    try:
        response = _get_cerebras_llm_with_tools(tool_choice=force_tool).invoke(messages)
    except Exception as exc:
        if _is_context_overflow_error(exc):
            raise ContextOverflowError(str(exc)) from exc
        raise
    _log_forced_result(decision, response)
    return _agent_result(response)


# record_* 도구 이름 → 그 인자가 쌓일 AgentState 채널. 새 도메인의 record_*
# 도구를 추가할 땐 이 매핑에 한 줄만 추가하면 된다 - 예전엔 _agent_result
# 안에 if/elif 분기를 손으로 늘려야 했는데, 한 번 빠뜨려서(record_food_facts
# 추가 때) 도구는 호출되는데 상태엔 하나도 안 남아 classify가 영영 안 걸리는
# 버그를 실제로 겪었다. 매핑 하나로 통일해서 이 버그 종류 자체를 없앤다.
_FACT_TOOL_TO_STATE_KEY = {
    "record_case_facts": "case_facts",
    "record_permit_synthesis": "permit_synthesis",
    "record_food_facts": "food_facts",
    "record_fire_facts": "fire_facts",
    "record_signage_facts": "signage_facts",
    "record_task_progress": "task_progress",
    # record_case_facts엔 address 필드가 없어서(법령 판정에 안 쓰임)
    # case_facts에 주소가 저장된 적이 없었다 - 로드맵 Step1의 "사업 예정지
    # 주소 확인" substep(src/roadmap_model.py)이 case_facts.get("address")를
    # 보는데 아무도 안 채워서 영원히 미완료로 남는 버그였다(2026-08-05,
    # 세션 b846cc22에서 신고). lookup_land_zone/lookup_building_ledger는
    # 이미 address를 인자로 받는 도구라, 호출될 때 그 값을 case_facts에도
    # 같이 적재한다 - 새 도구ᆞ새 판정 로직 추가 없이 이미 있는 호출에서
    # 얻는 부수 효과.
    "lookup_land_zone": "case_facts",
    "lookup_building_ledger": "case_facts",
}


def _agent_result(response: AIMessage) -> dict:
    """LLM 응답에서 _FACT_TOOL_TO_STATE_KEY에 등록된 record_* 호출을 찾아
    각각 대응하는 상태 채널 갱신분으로 뽑아낸다.

    도구 자체는 여전히 ToolNode가 정상 실행해서(확인 문자열만 반환) 도구
    호출-응답 짝은 그대로 맞춰지고, 여기서는 그 인자를 그래프 상태에도
    반영하는 부수 작업만 한다.
    """
    updates: dict[str, dict] = {}
    for tc in getattr(response, "tool_calls", None) or []:
        state_key = _FACT_TOOL_TO_STATE_KEY.get(tc["name"])
        if state_key is not None:
            updates.setdefault(state_key, {}).update(_coerce_tool_args(tc["name"], tc["args"]))
    if "task_progress" in updates:
        updates["task_progress"] = cascade_construction_progress(updates["task_progress"])
    return {"messages": [response], **updates}


_TOOL_ARGS_SCHEMA = {t.name: t.args_schema for t in TOOLS}


def _coerce_tool_args(tool_name: str, args: dict) -> dict:
    """LLM이 보낸 raw args를 그 도구의 pydantic 스키마로 통과시켜 타입을 맞춘다.

    ToolNode가 도구를 **실행**할 때는 이 검증을 거치지만, 여기서는 tc["args"]를
    날것으로 읽어 상태에 넣기 때문에 그 변환을 우회하고 있었다. 모델마다 JSON
    직렬화가 달라서 실제로 문제가 됐다(2026-08-02 실측):

        Gemini : {'floors': 3,   'current_facility_group': 8}    ← int
        Gemma4 : {'floors': '3', 'current_facility_group': '8'}  ← 문자열

    Gemma4는 값을 틀린 게 아니라(사무실→8, 카페→7 매핑까지 전부 정확) 타입만
    문자열로 보냈다. 그런데 그게 그대로 상태에 들어가면 classify_case에서
    `facts["floors"] < 3`이 **TypeError로 크래시**한다(신축ᆞ대수선 경로).
    용도변경은 시설군이 1~9 한 자리라 문자열 비교가 우연히 숫자 순서와 같아
    "동작하는 것처럼" 보였는데, 이것도 운이지 설계가 아니다.

    프롬프트로 "숫자로 보내라"고 지시하는 것보다 여기서 강제 변환하는 게 맞다 -
    모델ᆞ프로바이더가 바뀌어도 유효하고, 스키마라는 이미 있는 단일 진실을
    재사용하기 때문이다. 검증 실패 시엔 원본을 그대로 두어 기존 동작을 유지한다
    (여기서 예외를 던지면 대화 자체가 끊긴다).
    """
    schema = _TOOL_ARGS_SCHEMA.get(tool_name)
    if schema is None or not args:
        return args
    try:
        validated = schema.model_validate(args).model_dump()
    except Exception as exc:  # ValidationError 등 - 원본 유지가 더 안전
        logger.warning("[coerce] %s args 검증 실패, 원본 사용: %s", tool_name, exc)
        return args
    # 원래 보낸 키만 남긴다 - model_dump()는 안 보낸 필드도 None으로 채워 돌려준다.
    return {k: validated[k] for k in args if k in validated}


# === 판정 분류 파이프라인(src/classify.py로 분리, Phase 3) ===
from src.classify import DOMAIN_CONFIGS  # noqa: E402


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
    유도하는 [Step 1-2]ᆞ[Step 1-3]이 서로 다른 턴에 걸쳐 나올 수 있어서
    하나만으로는 부족함) 그 결과를 최종 스냅샷으로 잠가서 이후의 무관한
    대화가 덮어쓰지 못하게 한다. related_laws는 여전히 잠금 신호로 안 쓴다
    (2026-07-24 실측 확인, 위 3번 참고).
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
    result.related_agencies = synthesis.get("related_agencies", [])
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





# === 라우팅 결정은 src/routing.py로 분리(Phase 5) ===
# route_after_agent/route_after_tools/route_after_guard는 이 파일에서 더 안
# 쓴다 - 이 파일이 src.routing을 import하면(routing.py가 add_conditional_edges
# 대상 함수라 AgentState를 실제로 import해야 해서) 순환참조가 되므로, 필요하면
# from src.routing import route_after_agent, ...로 직접 가져올 것.


# === ToolNode 생성 ===
tool_node = ToolNode(tools=TOOLS)   # tool 콜 받아서 실제 함수 실행


# === 그래프 조립(src/graph.py로 분리, Phase 2) ===
# build_graph()/graph는 agent_node 등 이 파일의 거의 모든 조각을 가져다
# 배선하므로, 반대 방향(이 파일이 src.graph를 import)은 순환참조가 된다.
# 필요하면 from src.graph import build_graph, graph로 가져올 것.
