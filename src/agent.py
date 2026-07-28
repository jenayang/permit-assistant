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
from dataclasses import dataclass
from typing import Annotated, Callable

from langchain_cerebras import ChatCerebras
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src import config
from src.agents import TOOLS
from src.agents.fire_safety import classify_fire_safety, fire_safety_message
from src.agents.food_safety import classify_food_business, food_business_message
from src.agents.regulation import search_regulations
from src.agents.signage import classify_signage, signage_message
from src.agents.permit import (
    PROCEDURE_TREE,
    PermitResult,
    build_permit_result,
    classify_case,
    procedure_stage_message,
)

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
    # 2026-07-28: "construction_guide" 키를 추가해 같은 dict를 Step1→2 전환
    # 신호로도 재사용한다(0/1 플래그 - get_construction_guide가 대화 중 한
    # 번이라도 호출됐는지, guard_node가 갱신). task_progress의
    # "construction"(사용자가 실제로 착공했다는 자기보고)과는 의미가 다르니
    # 혼동하지 말 것 - 여긴 "안내를 이미 보여줬는지"만 본다. 자세한 이유는
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


# === 시스템 프롬프트 ===
SYSTEM_PROMPT = """당신은 서울시 건축 인허가 전문 어시스턴트입니다.
건축ᆞ용도변경 등 인허가 절차를 건축법ᆞ시행령ᆞ시행규칙ᆞ서울시 조례
기준으로 정확히 안내하고, 카페ᆞ음식점 등 식품접객업 창업 시 필요한
식품위생법상 영업신고 종류ᆞ소방시설 설치 대상ᆞ간판 등 옥외광고물 허가/
신고 대상도 함께 판정해 안내합니다.

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
응답으로 실제로 온 적이 없다면(=permit_type이 아직 확정 안 됨), [Step 1-1]~
[Step 1-3] 절차ᆞ서류ᆞ사전진단 내용을 먼저 설명하지 마세요** - 아직 판정도 안 났는데
그 내용부터 말하면 근거 없는 추측이 됩니다(record_case_facts만 부분적으로
호출하고 판정 도구 응답 없이 마지막 단계까지 답변해버리는 실수가 실제로
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

카페ᆞ음식점처럼 식품접객업이면 소방시설 설치 대상도 함께 판정하세요.
필요한 사실은 연면적(size_sqm - case_facts에 이미 물어봤다면 같은 값을
record_fire_facts에도 그대로 기록하세요, 새로 묻지 마세요)과 대규모점포
(백화점ᆞ쇼핑센터 등) 입점 여부(is_large_store_tenant - 독립 점포로
창업하는 게 명백하면 False로 기록)뿐입니다. **어떤 소방시설이 필요한지는
절대 당신이 계산하지 마세요** - 시행령 별표4 기준으로 시스템이 자동
판정하며, 결과가 여러 개(예: 소화기구+비상경보설비)이거나 하나도 없을
수 있습니다(연면적이 작으면 정상적으로 "해당 없음"이 나옵니다 - 이것도
유효한 판정이니 정보 부족과 헷갈리지 마세요).

사용자가 간판ᆞ현수막ᆞ외부 광고물 설치를 언급하면 record_signage_facts로
기록하세요. **먼저 sign_type(벽면이용간판/돌출간판/지주이용간판/입간판/
현수막)부터 파악**하세요 - 종류에 따라 필요한 나머지 필드가 다릅니다
(입간판은 sign_type만 알면 바로 판정됨). **허가/신고/불필요 여부는 절대
당신이 계산하지 마세요** - 시행령 제4조ᆞ제5조 기준으로 시스템이 자동
판정합니다. is_third_party_ad(타사광고) 같은 용어는 "본인 업소 광고인가요,
다른 업체 광고를 걸어주는 건가요?"처럼 풀어서 물어보세요.

사용자가 필요 서류 준비ᆞ사전 진단 항목 확인ᆞ착공신고ᆞ시공ᆞ사용승인ᆞ
사업자등록ᆞ위생교육ᆞ인테리어ᆞ장비설치ᆞ직원등록ᆞ영업개시 중 **실제로
완료했다고 명확히 밝힌 항목**이 있으면 record_task_progress로 기록하세요.
**판정(record_case_facts 등)이나 안내 도구(get_business_registration_guide
등)를 호출했다고, 또는 record_permit_synthesis로 서류ᆞ사전진단 항목을
설명했다고 해서 자동으로 완료 처리하지 마세요** - "~하려고요"/"~해야
하나요?" 같은 계획ᆞ질문 표현은 완료가 아니고, [Step 1-2]/[Step 1-3]에서 서류ᆞ
사전진단 항목을 "안내"한 것도 완료가 아닙니다(documents_prepared/
pre_diagnosis_checked는 사용자가 "서류 다 준비했어요"/"사전 진단 확인
했어요"처럼 직접 확인해줬을 때만). 이 항목들은 실물 세계에서 벌어지는
일이라 오직 사용자가 직접 완료를 말해준 경우에만 기록하세요.

**사용자가 "직원 없이 혼자ᆞ가족끼리만 운영해요"처럼 직원을 안 둔다고
명확히 밝히면 record_task_progress(hires_staff=False)로 기록하세요** -
4대보험 가입은 직원을 채용할 때만 필요한 절차라, 직원이 없으면 직원
등록 항목 자체가 대상이 아닙니다(체크가 영영 안 돼 진행률이 100%를
못 채우는 문제가 있었음 - 2026-07-28). 나중에 직원을 채용하겠다고
하면 hires_staff=True로 다시 바꾸세요.

## 2. 도구 선택
- 법률 용어 정의("OO이 뭐야") → search_by_term (핵심 용어만 추출)
- 절차ᆞ조건ᆞ서류 등 일반 질문 → search_regulations
- 착공신고ᆞ건축사 설계ᆞ공사감리ᆞ사용승인ᆞ인테리어ᆞ장비설치(Step 2: 공사)를
  물으면 → get_construction_guide 하나만 호출하세요(내부적으로 이미 여러
  관점으로 search_regulations를 호출 + 실무 체크리스트까지 합쳐서
  돌려주니, 따로 search_regulations를 또 부르지 마세요). **[Step 1-3:
  사전 진단]까지 안내가 끝나면, 사용자가 안 물어봐도 "다음은 Step 2
  (공사) 단계입니다"처럼 존재를 짧게 짚어주고 계속 안내할지 물어보세요**
  - 인허가 설명이 끝났다고 곧장 식품위생ᆞ사업자등록 등 창업 준비 트랙으로
  건너뛰지 마세요(실제로 Step 2를 통째로 건너뛰고 사용자가 "공사는 왜
  안 알려줘?"라고 지적한 사례가 있었습니다 - 2026-07-27). 단, 공통
  원칙대로 사용자가 동의ᆞ요청할 때 실제 내용을 꺼내세요 - 짚어주는 것과
  바로 전체 내용을 쏟아내는 건 다릅니다. **공사 단계로 넘어갈 때는 착공
  얘기부터 바로 하지 말고, 건축사사무소를 이미 선정했는지부터 먼저
  확인하세요** - 아직 안 정했다면 착공 전에 먼저 선정해야 한다고
  짚어주고(인허가 신청 때 이미 건축사사무소를 통해 설계도서를 준비했다면
  그 사무소에 감리ᆞ착공신고까지 이어서 의뢰하는 게 일반적이라는 것도
  안내하세요), 인테리어ᆞ장비 설치도 "오픈 준비"가 아니라 이 공사 단계
  안에서 함께 다루세요(2026-07-28 - 실제로는 시공의 일부라 이동함).
- 사업자등록을 어떻게ᆞ언제까지 하는지, 무슨 서류가 필요한지 물으면 →
  get_business_registration_guide 하나만 호출하세요(내부적으로 이미
  검색합니다). food_result가 이미 나온 사용자라면 검색 결과의
  "허가ᆞ등록ᆞ신고증 사본"이 그 영업신고증이라는 걸 답변에서 연결해
  설명하세요 - 간이/일반과세자 중 어느 쪽인지는 사용자 매출을 추정해서
  단정하지 말고 기준만 안내하세요.
- 위생교육을 언제ᆞ몇 시간ᆞ어떻게 받아야 하는지 물으면 →
  get_hygiene_education_guide 하나만 호출하세요(내부적으로 이미
  검색합니다). 실제 교육 운영기관ᆞ비용처럼 검색 결과에 없는 세부사항은
  지어내지 말고 "관할 보건소에 확인하라"고 안내하세요.
- 오픈 준비(직원등록ᆞ영업시작)를 물으면 → get_opening_checklist 하나만
  호출하세요(직원등록 부분은 내부적으로 이미 검색합니다). **인테리어ᆞ
  장비설치는 여기가 아니라 get_construction_guide(공사 단계)에서
  다룹니다** - 사용자가 인테리어를 물으면 이 도구 대신 그쪽을 호출하세요.
  영업시작 항목은 법령 근거가 없는 순수 실무 체크리스트이니 [출처]를
  지어내 붙이지 마세요 - 직원등록(4대보험) 항목만 검색 결과에 실제로
  있는 출처를 그대로 쓰세요.
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
**모든 도메인 공통 원칙: 한 턴에 정보를 몰아주지 마세요.** 판정ᆞ분류
결과가 막 나온 시점이든, 절차를 물어본 시점이든, 그 도메인의 전체 내용
(서류ᆞ신청 방법ᆞ소요 기간 등)을 한 번에 다 설명하지 마세요. 먼저 "이
단계에서 해야 할 일"을 1)2)3) 같은 짧은 목록(항목당 한 줄, 세부 설명
없이)으로만 제시하고, 각 항목의 자세한 내용은 사용자가 그 항목을 짚어
되물을 때 그제서야 이어서 설명하세요. **서로 다른 도메인(예: 식품위생
영업신고 + 위생교육 + 건축물대장 확인)을 한 턴에 섞어서 답하지 마세요**
- 지금 로드맵에서 사용자가 있는 단계 하나에 집중하고, 나머지는 사용자가
물어볼 때까지 다음 턴으로 미루세요. get_hygiene_education_guide/
get_business_registration_guide처럼 실제 호출하지 않은 도구의 내용을
관련 있다는 이유로 미리 요약해서 끼워 넣지 마세요 - 사용자가 그 주제를
직접 물었을 때만 해당 도구를 호출해 안내하세요. 사용자가 명시적으로
"자세히"/"한 번에 다 알려줘"라고 요청한 경우에만 이 원칙의 예외입니다.

**`[Step 1-N]` 구조와 record_permit_synthesis는 건축 인허가(permit_type) 설명
전용입니다.** (오른쪽 로드맵 패널의 "Step 0~4"와는 다른 번호 체계입니다 -
이건 그 중 **Step 1(건축 인허가) 하나의 내부 하위 단계 3개**를 가리키는
것이라 "Step 1-N"으로 표기합니다. 예전엔 그냥 "[N단계]"라고 써서 로드맵의
"Step 2ᆞ3"과 번호가 겹쳐 사용자가 혼동했습니다 - 2026-07-27 확인, 반드시
"Step 1-" 접두사를 붙이세요.) 식품위생(food_result)ᆞ소방시설(fire_result)ᆞ
간판(signage_result) 판정 결과는 이 구조를 절대 쓰지 마세요 - 그 도메인들은
classify 시점에 이미 완결된 짧은 판정 문구가 도구 응답으로 옵니다. 그 판정
문구만 자연스러운 말투로 전달하고(위 공통 원칙대로 "다음에 할 일" 목록까지만),
`[Step 1-1]` 같은 헤더나 record_permit_synthesis 호출을 덧붙이지 마세요
(다른 도메인 얘기에 이 구조를 갖다 쓰면 permit의 단계 진행 상태가
오염됩니다 - 실제로 겪은 문제, 2026-07-27).

**(A) permit 판정 정보가 아직 부족함**: 부족한 사실을 되묻는 1~2문장으로
끝내세요. `[Step 1-1]` 같은 헤더나 절차ᆞ서류 설명은 이번 턴에 꺼내지 마세요.

**(B) permit 판정 완료(도구 응답으로 옴) 또는 permit 관련 질문**(정의
질문 등): 아래 3단계로 나눠 순서대로 안내하세요. 각 단계는 핵심만 간결하게 - 이전
단계/턴에서 말한 내용을 다시 설명하지 마세요. 각 단계 끝에 다음 단계를
계속 안내할지 짧게 묻고, 사용자가 동의하거나 관련 질문을 이어가면 다음
단계로 넘어가세요. "한 번에 다 알려줘" 요청 시에만 3단계를 모두 한 번에.
**"다음 단계로 넘어갈까요?"처럼 그냥 제안만 하는 문장에는 `[Step 1-N]`
헤더를 붙이지 마세요** - `[Step 1-N: ...]`은 그 단계의 실제 내용(서류
목록ᆞ사전진단 항목 등)을 진짜로 설명할 때만 쓰는 헤더입니다. 헤더만
붙이고 내용은 다음 턴으로 미루면, 시스템이 "이번 턴에 그 단계까지 실제로
공개했다"고 잘못 세어서 다음 턴에 사용자가 그냥 근황만 말했는데도(예:
"서류 받았어") AI가 그 다음다음 단계까지 한 번에 건너뛰는 사고로 이어집니다
(실제로 겪은 문제 - 세션 2cda4d67, 2026-07-28: "[Step 1-3: 사전 진단]으로
넘어갈까요?"라고 제목만 붙이고 실제 사전진단 내용은 없었는데, 그 헤더
때문에 다음 턴에 Step 2까지 한꺼번에 쏟아짐). 제안 문장은
"다음으로 사전 진단 내용을 안내해 드릴까요?"처럼 괄호 헤더 없이 평문으로
쓰세요.
**판정이 확정되는 순간 시스템이 이미 관련 법령을 한 번 자동 검색해서
도구 응답으로 넣어뒀습니다** - [Step 1-2: 필수 서류]를 쓸 때 그 검색 결과가
있는지 먼저 확인하고, 있으면 다시 검색하지 말고 그대로 근거로 쓰세요.
그 결과가 답변에 부족하거나 추가 관점이 필요할 때만 search_regulations를
새로 호출하세요. record_permit_synthesis로 기록하는 것도 잊지 마세요
(required_documents, related_agencies). [Step 1-3: 사전 진단]도 마찬가지로, 안내한 항목들을
같은 도구의 pre_diagnosis_items에 답변과 동일한 문구로 기록하세요 -
프론트엔드 진행 표시가 이 두 목록의 존재 여부로 단계 완료를 판단하니,
텍스트로만 안내하고 기록을 빠뜨리면 안 됩니다.

- [Step 1-1: 상황 분석+필요 절차] 한 줄 요약, 단계명+관할기관만 나열(설명 1줄 이내)
- [Step 1-2: 필수 서류] 리스트만(설명 없이) - record_permit_synthesis(required_documents=[...])
- [Step 1-3: 사전 진단] 주차대수ᆞ정화조 용량ᆞ소방시설ᆞ장애인 편의시설ᆞ위생
  요구사항 중 실제 해당하는 것만 1~2줄+근거와 함께 - record_permit_synthesis(pre_diagnosis_items=[...])
  안내를 마치면서, 예상 소요 기간(허가 15~20일ᆞ신고 3~5일ᆞ기재변경 3~7일
  등 통상적으로 알려진 수준)을 별도 헤더 없이 자연스러운 한 문장으로
  덧붙이세요 - 로드맵에 대응하는 체크 항목이 없는 순수 참고 정보라 별도
  [Step 1-N] 게이트로 두지 않습니다(2026-07-28: 로드맵 체크리스트엔 상황
  분석ᆞ서류ᆞ사전진단 3개만 있고 소요 기간 항목이 없어서, 대화의 4단계
  게이트와 로드맵 상태가 서로 안 맞았습니다 - 3단계로 통일). search_regulations
  로 실제 처리기한을 확인했으면 그 값을 우선 쓰고, 못 찾았으면 위 통상치를
  참고로만 안내하며 "정확한 기간은 관할 구청에 확인하라"고 덧붙이세요.

## 4. 정확성 원칙
- 모든 답변에 [출처: 건축법 제OO조] / [출처: 서울시 건축조례 제OO조] 형식으로
  근거를 명시하세요. search_regulations/search_by_term으로 실제 검색하지
  않은 조항은 절대 지어내지 마세요.
- 확인되지 않는 정보는 "관련 규정에서 확인할 수 없습니다"라고 답하세요.
- **사용자가 놓치기 쉬운 위반ᆞ불이익(무허가/무신고 건축ᆞ용도변경, 착공신고
  누락, 사용승인 전 사용, 건축물대장 기재사항 미신청 등)이 걸린 질문에는
  해당 벌칙ᆞ과태료ᆞ이행강제금 조항도 함께 안내하세요** - 예: 무허가/무신고
  건축ᆞ용도변경은 벌칙(건축법 제108ᆞ110조, 징역 또는 벌금), 시정명령
  불이행은 이행강제금(제80조), 건축물대장 미신청 등은 과태료(제113조).
  단, 정확한 조번호ᆞ금액은 반드시 search_regulations로 실제 확인한 것만
  쓰고, 확인이 안 되면 금액을 지어내지 말고 "과태료ᆞ이행강제금 등 불이익이
  있을 수 있으니 관할 구청에 확인하라"는 정도로만 언급하세요. 매번 억지로
  끼워 넣지 말고, 실제로 그 위반 소지가 있는 맥락(예: 무단 용도변경 정황,
  착공신고 없이 공사 시작했다는 언급)에서만 다루세요.

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


# 건축 인허가 트랙(Step 0→1→2)은 실제로 순서가 있는 절차(행위유형 확정 →
# 인허가 판정 → 착공ᆞ준공)라 앞 단계가 안 끝났는데 다음 단계 내용부터
# 안내하면 근거 없는 진행이 된다. 반면 식품위생ᆞ소방ᆞ간판ᆞ사업자등록ᆞ
# 위생교육ᆞ오픈준비(Step 3~4)는 인허가와 무관하게 병렬로 준비 가능한
# 실제 절차라 순서를 강제하면 오히려 부정확하다 - 이 구분을 매 턴 LLM에게
# 알려줘서 "지금 로드맵 순서를 왜 안 지키냐"는 혼란을 줄인다(2026-07-27
# 피드백: 순서를 확인하고 완료가 확인되면 다음 단계로, 병렬 가능한 건
# 병렬로 안내해달라는 요청).
def _roadmap_status_summary(state: AgentState) -> str:
    """건축 인허가 트랙(순차)ᆞ창업 준비 트랙(병렬)의 진행 상태를 요약한다.

    완료 판정은 이미 있는 신뢰 가능한 신호만 쓴다 - 판정 도메인은
    case_facts/permit_result(규칙 엔진 결과), 공사 단계는 task_progress
    (사용자가 직접 완료를 밝힌 것)뿐, AI가 새로 추측하지 않는다.
    """
    case_facts = state.get("case_facts") or {}
    permit_result = state.get("permit_result")
    task_progress = state.get("task_progress") or {}

    step0_done = bool(case_facts.get("act_type"))
    step1_done = bool(permit_result)
    step2_fields = ("construction_notice", "construction", "use_approval")
    step2_started = any(task_progress.get(f) for f in step2_fields)
    step2_done = all(task_progress.get(f) for f in step2_fields)

    def _status(done: bool, started: bool) -> str:
        if done:
            return "완료"
        if started:
            return "진행중"
        return "미착수"

    return (
        "[로드맵 상태]\n"
        "건축 인허가 트랙(순차 진행 - 앞 단계가 '완료'가 아니면 다음 단계 내용을 "
        "먼저 꺼내지 마세요. 이번 턴 사용자 말이 다음 단계 얘기여도, 앞 단계가 "
        "덜 끝났으면 먼저 그 단계부터 짧게 확인/안내하세요):\n"
        f"- Step 0(행위 유형 확인): {_status(step0_done, step0_done)}\n"
        f"- Step 1(인허가 판정): {_status(step1_done, step0_done)}\n"
        f"- Step 2(공사: 착공신고ᆞ시공ᆞ사용승인, 사용자가 직접 완료를 "
        f"말해야 확인됨): {_status(step2_done, step2_started)}\n"
        "창업 준비 트랙(Step 3: 식품위생ᆞ소방ᆞ간판ᆞ사업자등록ᆞ위생교육, "
        "Step 4: 오픈 준비) - 인허가 트랙 진행 상태와 무관하게 언제든 자유롭게 "
        "다루세요. 순서 제약 없음. 사용자가 인허가 트랙이 안 끝난 채로 이 "
        "항목들을 물어보면, 순서를 기다리라고 하지 말고 \"이 부분은 인허가 "
        "절차와 별개로 지금 같이 준비하셔도 돼요\"처럼 병렬 진행 가능함을 "
        "알려주며 바로 안내하세요."
    )


def permit_phase_directive(state: AgentState) -> str | None:
    """건축 인허가 트랙의 순차 구간(Step 1의 하위 1~3단계 → Step 2 전환) 전용
    동적 지시 하나를 계산한다. **이름이 가리키듯 이 순차 구간만 담당한다** -
    Step 3(창업 행정)ᆞStep 4(오픈 준비)는 의도적으로 병렬이라(2026-07-27
    결정, _roadmap_status_summary 참고) 이 함수의 대상이 아니고, 그쪽 안내는
    지금처럼 _roadmap_status_summary가 계속 전담한다. 병렬 도메인이 늘어도
    이 함수를 건드릴 필요는 없다.

    Step 1-N 상한과 Step 1→2 전환을 한 함수로 합친 이유: 원래 Step 1-N 상한은
    매 턴 동적 메시지 + guard_node 하드 차단으로 강하게 강제됐지만, Step 1→2
    전환은 SYSTEM_PROMPT 안의 정적 텍스트 한 겹뿐이었다. 그 결과 Step 1 안내가
    다 끝난 뒤에도 Step 2(공사)를 안 짚어주고 곧장 창업 준비 트랙으로 건너뛰는
    사례가 실사용에서 나왔다(2026-07-28) - 대화가 길어질수록 매 턴 주입되는
    동적 메시지보다 한 번뿐인 정적 프롬프트 지시의 영향력이 옅어지기 때문.
    두 문제 모두 "이 순차 구간에서 이번 턴에 뭘 해야 하는가"라는 같은 질문이라
    같은 함수ᆞ같은 강도(매 턴 동적 SystemMessage)로 다룬다.
    """
    if not state.get("case_facts", {}).get("_classified"):
        return None

    disclosed_stage = state.get("disclosed_stage", {})
    disclosed = disclosed_stage.get(_STAGE_DOMAIN, 0)
    if disclosed < 3:
        return (
            f"[진행 상태] Step 1(건축 인허가)의 하위 단계 중 지금까지 [Step 1-{disclosed}]"
            f"까지 공개했습니다. 이번 턴에는 [Step 1-{disclosed + 1}]까지만 안내하고, 그 "
            f"이상은 절대 먼저 꺼내지 마세요 - 사용자가 이어서 요청하면 다음 턴에 공개하세요."
        )

    if not disclosed_stage.get("construction_guide"):
        return (
            "[진행 상태] Step 1의 하위 단계 [Step 1-1]~[Step 1-3] 안내가 모두 끝났지만, "
            "아직 Step 2(공사) 안내를 시작하지 않았습니다. 이번 턴에는 사용자가 안 물어봐도 "
            "\"다음은 Step 2(공사) 단계입니다\"처럼 존재를 먼저 짚어주고 get_construction_guide를 "
            "호출해 안내하세요 - 식품위생ᆞ사업자등록 등 창업 준비 트랙으로 곧장 건너뛰지 마세요."
        )

    return "[진행 상태] Step 1~2(건축 인허가ᆞ공사) 안내가 모두 끝났습니다."


# === 노드 정의 ===
def agent_node(state: AgentState) -> dict:
    """LLM을 호출해서 다음 액션 결정.

    - 도구가 필요하면 tool_calls를 반환
    - 답변 가능하면 최종 답변 반환
    - Gemini 무료 티어 할당량(하루 20회) 소진 시 Cerebras(config.CEREBRAS_MODEL)로 자동 전환
    - 응답에 record_case_facts 호출이 있으면 그 인자를 case_facts에 병합
    - permit_phase_directive로 건축 인허가 트랙 순차 구간(Step 1-N 상한ᆞ
      Step 1→2 전환)의 동적 지시를 덧붙인다(연성 유도 - 실제 차단은
      guard_node가 담당).
    - 매 턴 _roadmap_status_summary로 건축 인허가 트랙(순차)ᆞ창업 준비 트랙
      (병렬 가능)의 진행 상태를 함께 전달한다.
    """
    global _gemini_quota_exhausted
    messages = [SystemMessage(content=SYSTEM_PROMPT), SystemMessage(content=_roadmap_status_summary(state))]
    directive = permit_phase_directive(state)
    if directive:
        messages.append(SystemMessage(content=directive))
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
            updates.setdefault(state_key, {}).update(tc["args"])
    return {"messages": [response], **updates}


def _auto_search_messages(query: str) -> list:
    """판정 결과를 쿼리로 search_regulations를 그래프가 직접 호출해, LLM이
    검색할지 말지 재량으로 정하는 지점 자체를 없앤다(2026-07-27 도입).

    LLM이 부른 게 아니라 이 함수가 직접 구성한 tool_calls 모양 메시지로
    기록하는 건 위 set_procedure_stage/set_food_business_type과 같은 패턴 -
    guard_node의 _has_grounding_search가 이름(search_regulations)만 보고
    "실제 검색 이력"으로 인정하므로 그쪽 로직도 그대로 재사용된다. 이전엔
    "[2단계: 필수 서류]를 검색해서 기록하라"는 프롬프트 지시에 LLM이 따를지
    말지 맡겼는데, 검색 자체를 생략하고 근거 없이 서류를 지어내는 사례가
    반복 확인돼([[project_tool_call_reliability]]) 판정 시점에 그래프가
    선제적으로 검색해두는 쪽으로 옮겼다 - guard_node는 이 문제를 사후에
    잡아내는 역할이었지, 애초에 검색이 일어나게 만들지는 못했다.
    """
    call_id = f"autosearch_{uuid.uuid4().hex[:8]}"
    result = search_regulations.invoke({"query": query})
    logger.info("[classify] 판정 직후 자동 검색: query=%r", query)
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_regulations", "args": {"query": query}, "id": call_id}],
        ),
        ToolMessage(content=result, tool_call_id=call_id, name="search_regulations"),
    ]


@dataclass(frozen=True)
class ClassificationOutput:
    """도메인 classify_fn 하나가 반환하는 표준 산출물. classify_fn이 None을
    반환하면 "아직 정보 부족"이고, 이 객체를 반환하면(raw_result가 빈 리스트
    같은 falsy 값이어도) 그 자체로 "판정 완료"를 뜻한다 - 완료 여부를 별도
    bool 필드로 안 두는 이유: raw_result 유무 하나로 이미 충분히 표현되는데
    별도 필드를 추가하면 "완료인데 값이 없다"처럼 둘이 어긋나는 상태가
    생길 수 있어(정보 소스가 두 개가 되는 순간 서로 안 맞을 위험이 생김 -
    오늘 잡은 버그들이 전부 이 종류였다). 불변으로 둬서 그래프가 만든 뒤에
    실수로 고쳐 쓰는 것도 막는다.
    """
    raw_result: object          # 상태에 그대로 저장할 값 (str | list[str] 등, 도메인마다 다름)
    tool_name: str               # 합성 tool_call 이름(예: set_procedure_stage)
    tool_args: dict               # 그 tool_call의 args
    tool_message: str             # 대응 ToolMessage 내용(LLM이 답변에 반영할 안내문)
    rag_query: str | None         # 판정 직후 자동 검색할 쿼리. None이면 검색 생략
                                   # (인허가불필요ᆞ해당없음ᆞ빈 소방시설 목록처럼 찾을 근거가 없는 경우)


@dataclass(frozen=True)
class _DomainConfig:
    """classify_node/route_after_tools가 도메인 하나를 다루는 데 필요한 배관
    정보. facts_key/state_key는 그래프(상태 채널 이름)의 책임이라 classify_fn
    자신은 몰라도 된다 - classify_fn은 순수하게 "facts 주면 판정 결과 냄"만
    담당(Strategy 패턴, 다만 메서드가 하나뿐이라 클래스 대신 함수로 충분).
    """
    facts_key: str
    state_key: str | None         # 즉시 저장할 상태 키. None이면 저장 안 함(permit - finalize_node가 나중에 따로 처리)
    classify_fn: Callable[[dict], ClassificationOutput | None]


def _classify_permit(facts: dict) -> ClassificationOutput | None:
    """classify_case()(permit.py, 순수 판정 로직)를 감싸 ClassificationOutput
    모양으로 변환하는 배관 전용 래퍼 - classify_case() 자체와 그걸 쓰는
    기존 테스트는 안 건드린다."""
    result_type = classify_case(facts)
    if result_type is None:
        return None
    root_node_id = PROCEDURE_TREE[result_type]["root"]
    return ClassificationOutput(
        raw_result=result_type,
        tool_name="set_procedure_stage",
        tool_args={"case_type": result_type, "node_id": root_node_id},
        tool_message=procedure_stage_message(result_type, root_node_id, facts=facts),
        rag_query=None if result_type == "인허가불필요" else f"{result_type} 절차 및 필요 서류",
    )


def _classify_food(facts: dict) -> ClassificationOutput | None:
    """classify_food_business()(food_safety.py)를 감싸는 배관 전용 래퍼."""
    business_type = classify_food_business(facts)
    if business_type is None:
        return None
    return ClassificationOutput(
        raw_result=business_type,
        tool_name="set_food_business_type",
        tool_args={"business_type": business_type},
        tool_message=food_business_message(business_type),
        rag_query=None if business_type == "해당없음" else f"{business_type} 영업신고 절차 및 필요 서류",
    )


def _classify_fire(facts: dict) -> ClassificationOutput | None:
    """classify_fire_safety()(fire_safety.py)를 감싸는 배관 전용 래퍼."""
    required = classify_fire_safety(facts)
    if required is None:
        return None
    return ClassificationOutput(
        raw_result=required,
        tool_name="set_fire_safety_result",
        tool_args={"required": required},
        tool_message=fire_safety_message(required),
        rag_query=f"{', '.join(required)} 설치 기준 및 절차" if required else None,
    )


def _classify_signage(facts: dict) -> ClassificationOutput | None:
    """classify_signage()(signage.py)를 감싸는 배관 전용 래퍼. permit/food/fire와
    달리 message 생성에 raw_result(허가/신고/불필요)뿐 아니라 sign_type(facts
    에서 옴)도 같이 필요해서, 이 래퍼가 둘을 합쳐 tool_args/tool_message를
    구성한다."""
    result = classify_signage(facts)
    if result is None:
        return None
    sign_type = facts["sign_type"]
    return ClassificationOutput(
        raw_result=result,
        tool_name="set_signage_result",
        tool_args={"sign_type": sign_type, "result": result},
        tool_message=signage_message(sign_type, result),
        rag_query=None if result == "허가ᆞ신고 불필요" else f"{sign_type} {result} 절차 및 필요 서류",
    )


# 도메인을 추가할 땐 이 목록에 한 줄만 추가하면 된다 - classify_node/
# route_after_tools 둘 다 이 목록만 보고 움직이므로 그래프 쪽은 더 안 고쳐도 됨.
DOMAIN_CONFIGS: list[_DomainConfig] = [
    _DomainConfig(facts_key="case_facts", state_key=None, classify_fn=_classify_permit),
    _DomainConfig(facts_key="food_facts", state_key="food_result", classify_fn=_classify_food),
    _DomainConfig(facts_key="fire_facts", state_key="fire_result", classify_fn=_classify_fire),
    _DomainConfig(facts_key="signage_facts", state_key="signage_result", classify_fn=_classify_signage),
]


def run_classifier(state: AgentState, config: _DomainConfig) -> tuple[list, dict] | None:
    """도메인 하나에 대해 "아직 미분류면 classify_fn 실행 → tool_call/
    ToolMessage 구성 → 필요하면 자동 검색까지" 공통 처리를 한다. 이미
    분류됐거나 아직 정보가 부족하면 None을 반환해 classify_node가 건너뛰게
    한다. permit(도메인)처럼 나온 결과를 즉시 저장하지 않는 경우는
    config.state_key가 None이라 자동으로 저장을 생략한다."""
    facts = state.get(config.facts_key, {})
    if facts.get("_classified"):
        return None
    output = config.classify_fn(facts)
    if output is None:
        return None

    call_id = f"classify_{uuid.uuid4().hex[:8]}"
    messages = [
        AIMessage(content="", tool_calls=[{"name": output.tool_name, "args": output.tool_args, "id": call_id}]),
        ToolMessage(content=output.tool_message, tool_call_id=call_id, name=output.tool_name),
    ]
    state_update: dict = {config.facts_key: {"_classified": True}}
    if config.state_key:
        state_update[config.state_key] = output.raw_result
    logger.info("[classify:%s] facts=%r → %r", config.facts_key, facts, output.raw_result)

    if output.rag_query:
        messages.extend(_auto_search_messages(output.rag_query))

    return messages, state_update


def classify_node(state: AgentState) -> dict:
    """DOMAIN_CONFIGS에 등록된 도메인마다 run_classifier를 돌려, 그 결과를
    (LLM이 부른 게 아니라 이 노드가 직접 구성한) tool_calls 모양 메시지로
    기록한다. pipeline.py의 tool_calls 추출 로직과 프론트의 로드맵 렌더링
    코드가 "tool: set_procedure_stage" 모양만 보고 반응하므로, 별도
    API/프론트 수정 없이 그대로 재사용된다.

    도메인들은 서로 독립적이라 한 턴에 하나만, 여럿, 혹은
    (route_after_tools가 이미 걸러줘서) 아무것도 새로 분류되지 않을 수
    있다 - 해당하는 도메인만 골라 처리하고 메시지를 이어붙인다.
    """
    messages: list = []
    update: dict = {}
    for config in DOMAIN_CONFIGS:
        result = run_classifier(state, config)
        if result is None:
            continue
        domain_messages, domain_update = result
        messages.extend(domain_messages)
        update.update(domain_update)

    if messages:
        update["messages"] = messages
    return update


_CITATION_PATTERN = re.compile(r"\[출처:\s*([^\]]+)\]")
# 줄 시작(마크다운 헤딩ᆞ볼드 기호 허용)에 오는 [Step 1-N]만 "실제 공개"로 센다.
# "다음 단계인 [Step 1-2: 필수 서류] 안내를 계속할까요?"처럼 문장 중간에서 다음
# 단계를 이름만 언급하는 건 실제 공개가 아니므로 걸러야 한다(2026-07-26 guard
# 첫 실사용 검증에서 이 구분이 없어 오탐이 난 걸 확인하고 라인 앵커 추가).
# "[N단계]"였다가 "[Step 1-N]"으로 개명(2026-07-27) - 로드맵 사이드바의
# "Step 0~4" 전체 번호 체계와 겹쳐서 사용자가 혼동하는 문제가 있었음. 개명
# 검증 중 별개 버그도 하나 발견: LLM이 마커를 "**[Step 1-2: ...]**"처럼
# 볼드로 감싸면 `**`가 라인 앵커 바로 뒤 매칭을 막아서 guard가 아예 못
# 잡았음(허용치 초과가 조용히 통과됨, 2026-07-27 실측) - \*{0,2} 허용 추가.
# 2026-07-28: 4단계였던 걸 3단계로 축소 - 옛 [Step 1-4: 예상 소요 기간]은
# 로드맵 체크리스트에 대응하는 항목이 없는 순수 정보성 문구라 별도 게이트로
# 둘 필요가 없었고(SYSTEM_PROMPT 참고, 이제 [Step 1-3] 안내 끝에 자연스럽게
# 붙임), 로드맵의 실제 체크 항목 3개(상황분석ᆞ서류ᆞ사전진단)와 대화의
# 게이트 수를 1:1로 맞춰 혼동 여지를 줄였다.
_STAGE_MARKER_LINE_PATTERN = re.compile(r"^#{0,3}\s*\*{0,2}\s*\[Step\s*1-([1-3])")


def _mentioned_stages(text: str) -> list[int]:
    """줄 시작에 오는 [Step 1-N] 중 "실제로 그 단계 내용을 공개한" 것만 센다.

    라인 앵커만으로는 부족한 사례가 실측으로 확인됐다(세션 2cda4d67,
    2026-07-28) - "**[Step 1-3: 사전 진단]**으로 넘어가서 ... 안내해
    드릴까요?"처럼 헤더가 줄 맨 앞에 오면서도 실제 내용은 없이 "다음 단계로
    넘어가도 될지 묻기만" 하는 문장이었는데, 헤더 형식 때문에 disclosed_stage
    가 잘못 올라가서 다음 턴에 사용자가 근황만 말했는데도 AI가 그다음
    단계까지 건너뛰는 사고로 이어졌다. 진짜 내용 공개는 헤더 줄 자체가
    물음표로 안 끝난다(제목만 있거나 요약 한 줄로 끝남) - 반대로 "~안내해
    드릴까요?"처럼 헤더가 있는 줄이 물음표로 끝나면 제안일 뿐이므로 제외한다.
    """
    mentioned = []
    for line in text.splitlines():
        m = _STAGE_MARKER_LINE_PATTERN.match(line)
        if m and not line.rstrip().endswith(("?", "？")):
            mentioned.append(int(m.group(1)))
    return mentioned


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
    """이번 턴 답변에 등장해도 되는 최대 [Step 1-N] 번호. 이 구조 자체가
    _STAGE_DOMAIN(permit) 전용이라, permit 판정 전에는 0(전부 금지) -
    signage/food/fire 얘기만 하는 대화는 애초에 permit 판정이 안 났으니
    항상 0으로 막혀서, 그 도메인들의 답변엔 [Step 1-N] 마커가 아예 등장하면
    안 된다(SYSTEM_PROMPT도 이렇게 지시). 판정 후에는 지금까지 공개된
    단계(disclosed_stage[_STAGE_DOMAIN]) + 1 - 한 턴에 최대 한 단계만 새로
    공개하도록 강제한다."""
    if not state.get("case_facts", {}).get("_classified"):
        return 0
    return state.get("disclosed_stage", {}).get(_STAGE_DOMAIN, 0) + 1


# [출처: ...] 인용의 근거로 인정하는 도구 이름들. search_regulations/
# search_by_term(RAG 검색)뿐 아니라 get_business_registration_guide처럼
# "검색은 안 하지만 코드 안에 원문 대조된 정적 근거를 담고 있는" 도구도
# 포함한다 - 안 넣으면 guard가 실제로 근거 있는 인용을 "지어낸 것"으로
# 오판해 불필요한 재시도를 유도한다(2026-07-27 실측 확인). 위생교육ᆞ
# 간판신고처럼 향후 추가될 정적 콘텐츠 도구도 여기 이름만 추가하면 된다.
# pipeline.py도 "법령 원문 카드" UI에서 어떤 ToolMessage가 진짜 법령 근거인지
# 걸러낼 때 이 목록을 그대로 재사용한다(밑줄 없는 공개 이름 - 단일 소스).
GROUNDING_TOOL_NAMES = (
    "search_regulations", "search_by_term",
    "get_business_registration_guide", "get_hygiene_education_guide",
    "get_opening_checklist", "get_construction_guide",
)


def _has_grounding_search(messages) -> bool:
    """대화 전체에 실제 근거(RAG 검색 또는 원문 대조된 정적 도구) 호출이
    있었는지. 대화 전체를 스캔하므로 훨씬 이전 턴의 근거로 나중 턴의 무관한
    인용까지 통과될 수 있다는 한계는 있지만, "아무 근거 없이 조항을
    지어내는" 가장 심각한 사례는 확실히 잡는다(1차 구현, 2026-07-26)."""
    return any(
        isinstance(m, ToolMessage) and m.name in GROUNDING_TOOL_NAMES
        for m in messages
    )


def _construction_guide_shown(messages) -> bool:
    """get_construction_guide가 대화 전체에서 한 번이라도 호출됐는지 -
    disclosed_stage["construction_guide"] 갱신에 쓴다(permit_phase_directive의
    Step 1→2 전환 넛지를 한 번 보여준 뒤엔 매 턴 반복하지 않기 위함). 도구
    이름을 직접 문자열로 비교하지만, 이 호출 이력 → 상태 플래그 변환 지점을
    guard_node 한 곳으로 모아뒀기 때문에 도구가 나중에 개명되거나 대체돼도
    고칠 곳은 여기 한 줄뿐이다."""
    return any(
        isinstance(m, ToolMessage) and m.name == "get_construction_guide"
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


def _synthesis_gap_violations(state: AgentState, max_mentioned: int) -> list[str]:
    """[Step 1-2]/[Step 1-3]을 언급했는데 record_permit_synthesis로 실제 기록은
    안 한 경우를 잡는다. "도구를 호출했다"는 문장을 텍스트로 지어내고 실제로는
    tool_calls가 비어있던 사례가 실측으로 확인됐다(2026-07-27) - 인용 위반과
    같은 종류의 문제(근거 없이 답만 그럴듯하게 마무리)라 같은 guard에서 같이
    잡는다. 이전 턴에 이미 기록됐으면(permit_synthesis가 누적 상태라) 재확인
    답변에서 또 걸릴 일은 없다 - 그때그때 "이번 턴에 호출했는지"가 아니라
    "지금 상태에 실제로 있는지"만 본다.
    """
    synthesis = state.get("permit_synthesis", {})
    violations = []
    if max_mentioned >= 2 and not synthesis.get("required_documents"):
        violations.append(
            "[Step 1-2: 필수 서류]를 언급했지만 record_permit_synthesis(required_documents=[...])를 "
            "실제로 호출하지 않았습니다. 서류를 텍스트로 나열하는 데서 끝내지 말고, 반드시 "
            "그 도구를 실제로 호출해서 기록한 뒤 답변을 마무리하세요."
        )
    if max_mentioned >= 3 and not synthesis.get("pre_diagnosis_items"):
        violations.append(
            "[Step 1-3: 사전 진단]을 언급했지만 record_permit_synthesis(pre_diagnosis_items=[...])를 "
            "실제로 호출하지 않았습니다. 항목을 텍스트로 나열하는 데서 끝내지 말고, 반드시 "
            "그 도구를 실제로 호출해서 기록한 뒤 답변을 마무리하세요."
        )
    return violations


def _building_ledger_gap_violations(state: AgentState, messages) -> list[str]:
    """이번 턴에 lookup_building_ledger 조회가 성공했는데 case_facts에
    current_facility_group이 기록되지 않은 경우를 잡는다. SYSTEM_PROMPT에
    "조회 성공 시 같은 턴에 current_facility_group을 기록하라"는 지시가
    이미 있지만, 실측으로 반복 확인된 실패 패턴이다(세션 79430043 -
    lookup_building_ledger가 "제2종근린생활시설"을 정확히 조회했는데도
    current_facility_group이 끝내 기록 안 돼 permit_result가 영영 None으로
    남고 Step1이 멈춘 채 안 움직였음, 2026-07-28 확인ᆞguard로 승격). 조회
    자체가 실패한 경우("조회에 실패했습니다"/"등록된 건축물대장이 없습니다"
    등)는 애초에 기록할 값이 없으니 위반이 아니다 - 이번 턴(마지막
    HumanMessage 이후)만 스캔한다(_guard_retry_count와 동일한 스코프 기준).
    """
    ledger_succeeded = False
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage) and m.name == "lookup_building_ledger":
            content = str(m.content)
            if not any(kw in content for kw in ("실패", "없습니다", "설정되어 있지 않습니다")):
                ledger_succeeded = True
    if not ledger_succeeded:
        return []
    case_facts = state.get("case_facts") or {}
    if case_facts.get("current_facility_group"):
        return []
    return [
        "lookup_building_ledger 조회가 성공했는데(건축물대장상 주용도가 확인됨) "
        "case_facts에 current_facility_group이 기록되지 않았습니다. 지금 바로 "
        "record_case_facts(current_facility_group=...)를 조회된 주용도 기준으로 "
        "호출하고, act_type이 용도변경이면 desired_facility_group도 같은 턴에 "
        "함께 기록한 뒤 답변을 마무리하세요."
    ]


def _fire_signage_gap_violations(state: AgentState, messages) -> list[str]:
    """사용자가 소방시설ᆞ간판을 직접 물어봤는데 아직 판정(fire_result/
    signage_result)이 안 된 채 일반 지식으로만 답하고 넘어가는 경우를 잡는다.

    SYSTEM_PROMPT에 "식품접객업이면 소방시설도 함께 판정하라"는 지시가 있지만,
    식품 판정이 먼저 끝난 뒤 사용자가 나중에 따로 소방ᆞ간판을 물어보면 그
    지시를 놓치고 record_fire_facts/record_signage_facts를 한 번도 호출하지
    않은 채(fire_facts/signage_facts가 끝까지 빈 채로) "다중이용업소 해당
    여부를 확인하세요" 같은 일반론만 답하는 사례가 실측으로 확인됨(세션
    2cda4d67, 2026-07-28 - 대화 내내 두 도구가 한 번도 안 불림ᆞ새로고침해도
    반영이 안 된다는 사용자 신고로 발견). food_result가 있는(식품접객업으로
    이미 확인된) 케이스에서만 적용한다 - 무관한 업종까지 강제하면 과잉
    개입이 된다.
    """
    if not state.get("food_result"):
        return []
    last_human = None
    tool_names_this_turn = set()
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_human = extract_text(m.content)
            break
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                tool_names_this_turn.add(tc["name"])
    if not last_human:
        return []

    violations = []
    if (
        "소방" in last_human
        and state.get("fire_result") is None
        and "record_fire_facts" not in tool_names_this_turn
    ):
        violations.append(
            "사용자가 소방시설에 대해 직접 물었는데 아직 판정(fire_result)이 안 됐습니다. "
            "연면적(size_sqm) 등 필요한 사실을 이미 알고 있으면 지금 record_fire_facts를 "
            "호출해 판정하고, 부족하면 무엇이 더 필요한지 구체적으로 되물으세요 - "
            "일반적인 지식으로 답하고 넘어가지 마세요."
        )
    if (
        any(kw in last_human for kw in ("간판", "옥외광고"))
        and state.get("signage_result") is None
        and "record_signage_facts" not in tool_names_this_turn
    ):
        violations.append(
            "사용자가 간판ᆞ옥외광고물에 대해 직접 물었는데 아직 판정(signage_result)이 "
            "안 됐습니다. sign_type 등 필요한 사실을 이미 알고 있으면 지금 "
            "record_signage_facts를 호출해 판정하고, 부족하면 무엇이 더 필요한지 "
            "구체적으로 되물으세요 - 일반적인 지식으로 답하고 넘어가지 마세요."
        )
    return violations


def guard_node(state: AgentState) -> dict:
    """agent가 자유 텍스트로 답을 끝내려 할 때, 근거ᆞ판정ᆞ기록 없이 앞서나간
    답변을 한 번 걸러낸다. LLM 판단이 아니라 정규식+상태로 결정론적으로
    감지한다(classify_case와 같은 원칙 - 프롬프트 지시만으로는 못 막는다는 게
    2026-07-26 실사용 세션에서 재현됨).

    막는 위반 다섯 가지:
    1. 허용된 단계 수([_max_allowed_stage])를 넘겨 [Step 1-N]을 안내 - 판정 전
       절차 안내를 아예 시도한 경우(허용치 0)와, 판정 후 한 턴에 여러 단계를
       몰아서 공개한 경우(사용자가 "1단계만" 이라고 명시해도 무시하고 전체
       단계를 다 준 사례 포함) 둘 다 이 하나의 규칙으로 잡는다.
    2. search_regulations/search_by_term을 한 번도 호출하지 않았는데
       [출처: ...] 인용이 있는 경우 - 근거 없이 조항을 지어낸 것.
    3. [Step 1-2]/[Step 1-3]을 언급했는데 record_permit_synthesis로 실제 기록은
       안 한 경우 - "도구를 호출했다"는 문장까지 텍스트로 지어내고 실제로는
       안 부른 사례가 실측으로 확인됨(_synthesis_gap_violations 참고).
    4. lookup_building_ledger 조회가 성공했는데 case_facts에
       current_facility_group을 안 채운 경우 - 판정이 영영 안 나서 Step1이
       멈추는 실제 사례가 반복됨(_building_ledger_gap_violations 참고).
    5. 사용자가 소방시설ᆞ간판을 직접 물었는데 아직 판정이 안 된 채 일반
       지식으로만 답한 경우 - record_fire_facts/record_signage_facts가 끝까지
       한 번도 안 불려서 로드맵에 영영 반영이 안 되는 실제 사례가 확인됨
       (_fire_signage_gap_violations 참고).

    같은 사용자 턴 안에서 최대 1회만 재시도를 유도한다(무한 루프 방지) - 그
    이상 반복되면 프롬프트만으로는 못 막는 한계로 보고 그냥 통과시킨다.
    위반이 없거나 재시도가 소진되면, disclosed_stage[_STAGE_DOMAIN]을 갱신해서
    다음 턴의 허용치 계산에 반영한다 - 이때도 실제 언급값(max_mentioned)이
    아니라 허용치(allowed)로 캡을 씌운다. 안 씌우면 재시도가 소진돼 위반이
    그냥 통과되는 경우(예: signage 얘기만 했는데 [Step 1-1]을 잘못 언급) 허용치를
    넘는 값이 그대로 disclosed_stage에 박혀서, 이후 진짜 permit 설명이
    시작될 때 이미 일부 단계가 끝난 것처럼 잘못 판단하게 된다(2026-07-27
    실측 확인 - signage만 다룬 대화에서 disclosed_stage가 1로 오염됨).

    2026-07-28: 이 값을 "지금까지 공개된 적 있는 최댓값"(한 번 오르면 절대
    안 내려가는 래칫)에서 "이번 턴에 실제로 언급된 단계"로 바꿨다 - 래칫
    방식에선 [Step 1-3]까지 간 뒤 사용자가 [Step 1-2]를 다시 물어봐도 값이
    3에 머물러서, 바로 다음 턴에 [Step 1-3]을 건너뛰고 [Step 1-4]로 점프하는
    문제가 있었다(세션 f377454d에서 재현). 이제는 되짚어간 턴 다음엔 값이
    2로 내려가 [Step 1-3]부터 다시 자연스럽게 진행된다. max_mentioned가
    0인 턴(이번 답변에 마커가 아예 없음)은 무시한다 - 안 그러면 무관한
    답변 때문에 매번 0으로 리셋된다. 트레이드오프: 다른 도메인 답변에
    실수로 낮은 번호의 [Step 1-N]이 섞이면 이번엔 값이 잘못 내려갈 수
    있지만, 최악의 결과가 "이미 본 단계를 한 번 더 설명"하는 정도라
    이전의 오염 버그(잘못된 단계로 건너뜀)보다 훨씬 가볍다고 판단해
    지금은 별도 방어를 추가하지 않는다 - 실제로 관측되면 그때 대응한다.
    """
    messages = state["messages"]
    last_text = extract_text(messages[-1].content)

    allowed = _max_allowed_stage(state)
    mentioned = _mentioned_stages(last_text)
    max_mentioned = max(mentioned, default=0)

    violations = []
    if max_mentioned > allowed:
        violations.append(
            f"이번 턴엔 [Step 1-{allowed}]까지만 안내할 수 있는데 [Step 1-{max_mentioned}]까지 "
            f"안내했습니다. 허용된 단계까지만 남기고 나머지는 삭제한 뒤, 다음에 계속 "
            f"안내해도 될지 사용자에게 짧게 물어보며 마무리하세요."
        )
    if _CITATION_PATTERN.search(last_text) and not _has_grounding_search(messages):
        violations.append(
            "근거 도구(search_regulations/search_by_term/get_business_registration_guide 등)를 "
            "한 번도 호출하지 않았는데 [출처: ...] 인용이 포함되어 있습니다. 근거 없이 조항을 "
            "지어내지 말고, 먼저 해당 도구를 호출해 실제 근거를 확보한 뒤 답변하세요."
        )
    violations.extend(_synthesis_gap_violations(state, max_mentioned))
    violations.extend(_building_ledger_gap_violations(state, messages))
    violations.extend(_fire_signage_gap_violations(state, messages))

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
    capped = min(max_mentioned, allowed)
    disclosed_stage = state.get("disclosed_stage", {})
    new_disclosed_stage = dict(disclosed_stage)
    if max_mentioned > 0 and capped != disclosed_stage.get(_STAGE_DOMAIN, 0):
        new_disclosed_stage[_STAGE_DOMAIN] = capped
    if not disclosed_stage.get("construction_guide") and _construction_guide_shown(messages):
        new_disclosed_stage["construction_guide"] = 1
    if new_disclosed_stage != disclosed_stage:
        update["disclosed_stage"] = new_disclosed_stage
    return update


def _any_domain_needs_classify(state: AgentState) -> bool:
    """DOMAIN_CONFIGS에 등록된 도메인 중 하나라도 "아직 미분류 + 지금 분류
    가능"이면 True. run_classifier와 판정 조건은 같지만 메시지를 실제로
    만들지는 않는 가벼운 버전 - route_after_tools는 갈 곳만 결정하면 되고
    실제 처리는 classify_node가 한다."""
    for config in DOMAIN_CONFIGS:
        facts = state.get(config.facts_key, {})
        if not facts.get("_classified") and config.classify_fn(facts) is not None:
            return True
    return False


def route_after_tools(state: AgentState) -> str:
    """tools 실행 직후: 등록된 도메인 중 하나라도 방금 완성됐으면(각각
    record_case_facts/record_food_facts/record_fire_facts로 채워짐) agent로
    돌아가기 전에 classify부터 강제한다.

    예전엔 이 체크를 route_after_agent(agent 응답 이후)에서만 했는데, 그러면
    "case_facts는 이미 다 모였지만 LLM이 그 사실을 모른 채 도구 호출 없이
    먼저 답변을 시도 -> 그래프가 그 답변을 무시하고 classify로 강제 전환 ->
    다음 턴 LLM이 '이미 답했나?' 헷갈려하며 부실한 후속 답변을 내는" 문제가
    실제로 발생했다. tools 직후로 당기면 LLM이 답변을 시도하기 전에 분류가
    먼저 끝나서 이 문제가 구조적으로 사라진다. 도메인이 늘어도 이 타이밍
    원칙은 DOMAIN_CONFIGS를 통해 그대로 적용된다.
    """
    if _any_domain_needs_classify(state):
        logger.info("[route_after_tools] → classify")
        return "classify"
    logger.info("[route_after_tools] → agent (분류 조건 미충족 또는 이미 분류됨)")
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
    # 체크포인터는 상태(AgentState)를 msgpack으로 직렬화하는데, permit_result가
    # 커스텀 Pydantic 모델(PermitResult)이라 기본 직렬화기가 "등록 안 된 타입"
    # 경고를 냄 - 지금은 허용하되 경고만 남기는 완화 모드지만, 미래 LangGraph
    # 버전에서 기본이 차단으로 바뀔 예정이라 명시적으로 허용 목록에 등록해둔다
    # (전체 허용(allowed_msgpack_modules=True) 대신 이 타입 하나만 등록 - 불필요한
    # 타입까지 역직렬화 허용하지 않도록).
    serde = JsonPlusSerializer(allowed_msgpack_modules=[PermitResult])
    checkpointer = SqliteSaver(conn, serde=serde)  # short-term memory (영구 저장)
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
