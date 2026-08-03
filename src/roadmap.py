"""로드맵 단계 추적 + 진행 안내 (agent.py에서 분리, Phase 4).

건축 인허가 트랙(Step 0→1→2)의 순차 진행 상태(disclosed_stage)를 계산ᆞ안내하는
코드. guard.py가 이 모듈의 `_max_allowed_stage`ᆞ`_mentioned_stages`ᆞ
`_STAGE_DOMAIN`ᆞ`_STEP2_KEYWORDS`를 가져다 쓴다(단방향 - 이 모듈은 guard.py를
모른다).
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import re

from langchain_core.messages import AIMessage, HumanMessage

from src.message_utils import extract_text

if TYPE_CHECKING:
    from src.agent import AgentState

# disclosed_stage(dict[domain, stage])에서 지금 유일하게 4단계(대상판정/사전검토/
# 서류준비/신청접수) 구조를 쓰는 도메인의 키. permit만 이 패턴을 쓴다 - food/fire/
# signage 등 나머지는 classify 결과 자체가 완결된 답이라 단계 구조가 없다
# (2026-07-27 논의: 두 번째로 이 패턴이 필요한 도메인이 실제로 생기면 그때
# DOMAIN_CONFIGS에 supports_stage 같은 필드로 일반화 - 지금은 이르다).
_STAGE_DOMAIN = "case_facts"

# 줄 시작(마크다운 헤딩ᆞ볼드 기호 허용)에 오는 [Step 1-N]만 "실제 공개"로 센다.
# "다음 단계인 [Step 1-2: 필수 서류] 안내를 계속할까요?"처럼 문장 중간에서 다음
# 단계를 이름만 언급하는 건 실제 공개가 아니므로 걸러야 한다(2026-07-26 guard
# 첫 실사용 검증에서 이 구분이 없어 오탐이 난 걸 확인하고 라인 앵커 추가).
# "[N단계]"였다가 "[Step 1-N]"으로 개명(2026-07-27) - 로드맵 사이드바의
# "Step 0~4" 전체 번호 체계와 겹쳐서 사용자가 혼동하는 문제가 있었음. 개명
# 검증 중 별개 버그도 하나 발견: LLM이 마커를 "**[Step 1-2: ...]**"처럼
# 볼드로 감싸면 `**`가 라인 앵커 바로 뒤 매칭을 막아서 guard가 아예 못
# 잡았음(허용치 초과가 조용히 통과됨, 2026-07-27 실측) - \*{0,2} 허용 추가.
# 2026-08-03: 4단계로 확장 - 실제 업무 순서(대상판정→사전검토→서류준비→
# 신청접수)를 반영해 "신청서 제출"에 뭉쳐 있던 서류 준비(몇 주 걸림)와 접수
# (몇 분짜리 행위)를 분리하고, 사전진단을 서류 준비보다 앞으로 옮겼다.
_STAGE_MARKER_LINE_PATTERN = re.compile(r"^#{0,3}\s*\*{0,2}\s*\[Step\s*1-([1-4])")


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


def _max_allowed_stage(state: "AgentState") -> int:
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


# 건축 인허가 트랙(Step 0→1→2)은 실제로 순서가 있는 절차(행위유형 확정 →
# 인허가 판정 → 착공ᆞ준공)라 앞 단계가 안 끝났는데 다음 단계 내용부터
# 안내하면 근거 없는 진행이 된다. 반면 식품위생ᆞ소방ᆞ간판ᆞ사업자등록ᆞ
# 위생교육ᆞ오픈준비(Step 3~4)는 인허가와 무관하게 병렬로 준비 가능한
# 실제 절차라 순서를 강제하면 오히려 부정확하다 - 이 구분을 매 턴 LLM에게
# 알려줘서 "지금 로드맵 순서를 왜 안 지키냐"는 혼란을 줄인다(2026-07-27
# 피드백: 순서를 확인하고 완료가 확인되면 다음 단계로, 병렬 가능한 건
# 병렬로 안내해달라는 요청).
def _roadmap_status_summary(state: "AgentState") -> str:
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
        "Step 4: 오픈 준비) - 인허가 트랙과 병렬이지만 '아무거나 지금'이 아니라 "
        "실무 순서에 맞춰 안내하세요:\n"
        "- 허가/신고 종류 '판정ᆞ확인'(식품위생 영업 종류ᆞ소방 대상ᆞ간판)은 아무 때나 "
        "가능하니, 사실이 모이면 판정 결과만 짧게 확인해 알려주세요.\n"
        "- 하지만 '실제 진행'은 공사 일정에 맞춰 시점을 구분해 짚어주세요. 인허가 트랙이 "
        "아직 안 끝났으면 창업 준비 항목을 처음부터 쏟아내지 말고, 다음처럼 안내:\n"
        "  · 공사(Step 2) 전ᆞ중에 미리 해둘 수 있는 것: 위생교육 이수, 사업자등록, 각종 판정 확인\n"
        "  · 공사 완료(사용승인) 후에 하는 것: 식품위생 영업신고 접수, 소방시설ᆞ간판 설치, "
        "직원등록, 영업시작\n"
        "- 사용자가 특정 항목을 직접 물으면 그건 바로 안내해도 됩니다. 다만 묻지 않았는데 "
        "먼저 꺼낼 때는 위 시점 구분(지금 미리 할 것 / 공사 후 할 것)을 함께 알려주세요 - "
        "\"이건 공사 들어가면 그때 같이 진행하시면 돼요\"처럼.\n"
        "- '공사 완료 후에 하는 것' 항목(영업신고 접수ᆞ소방시설ᆞ간판 설치ᆞ직원등록ᆞ영업시작)을 "
        "안내하기 직전에는, 사용승인이 위 Step 2 상태에 아직 반영되지 않았으면 곧장 안내하지 "
        "말고 \"사용승인은 받으셨어요?\"처럼 완료 여부부터 확인하세요. 완료했다고 답하면 "
        "record_task_progress(use_approval=True)로 기록한 뒤 다음 턴부터 안내하세요."
    )


def permit_phase_directive(state: "AgentState") -> str | None:
    """건축 인허가 트랙의 순차 구간(Step 1의 하위 1~4단계 → Step 2 전환) 전용
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
    task_progress = state.get("task_progress") or {}

    # 사전검토(1-2)ᆞ서류준비(1-3) 안내가 끝나도 실제 완료 여부를 확인 안 받고
    # 그냥 다음 단계로 넘어가던 문제(2026-08-03) - Step 1-4→Step 2 전환에 이미
    # 있던 "완료 확인 후에만 다음 단계" 게이트를 같은 방식으로 확장한다.
    if disclosed == 2 and not task_progress.get("pre_diagnosis_checked"):
        return (
            "[진행 상태] Step 1-2(사전 검토) 안내까지 끝났습니다. 이번 턴에는 Step 1-3(설계ᆞ"
            "서류 준비)로 넘어가지 말고 \"사전 검토 항목은 확인해 보셨어요?\"처럼 완료 여부부터 "
            "확인하세요. 완료했다고 답하면 record_task_progress(pre_diagnosis_checked=True)로 "
            "기록한 뒤 다음 턴부터 Step 1-3을 안내하세요."
        )
    if disclosed == 3 and not task_progress.get("documents_prepared"):
        return (
            "[진행 상태] Step 1-3(설계ᆞ서류 준비) 안내까지 끝났습니다. 이번 턴에는 Step 1-4(신청ᆞ"
            "접수)로 넘어가지 말고 \"설계도서ᆞ서류 준비는 다 되셨어요?\"처럼 완료 여부부터 확인하세요. "
            "완료했다고 답하면 record_task_progress(documents_prepared=True)로 기록한 뒤 다음 "
            "턴부터 Step 1-4를 안내하세요."
        )
    if disclosed < 4:
        return (
            f"[진행 상태] Step 1(건축 인허가)의 하위 단계 중 지금까지 [Step 1-{disclosed}]"
            f"까지 공개했습니다. 이번 턴에는 [Step 1-{disclosed + 1}]까지만 안내하고, 그 "
            f"이상은 절대 먼저 꺼내지 마세요 - 사용자가 이어서 요청하면 다음 턴에 공개하세요."
        )

    if not disclosed_stage.get("construction_guide"):
        if not task_progress.get("application_submitted"):
            return (
                "[진행 상태] Step 1의 하위 단계 [Step 1-1]~[Step 1-4] 안내는 모두 끝났지만, "
                "신청ᆞ접수(Step 1-4에서 안내한 접수 절차)가 실제로 완료됐는지는 아직 확인되지 "
                "않았습니다. Step 2(공사)는 허가/신고 수리가 끝나야 실제로 진행할 수 있는 "
                "단계이니, 이번 턴에는 Step 2를 먼저 안내하지 말고 "
                "\"신청서 접수는 다 하셨어요?\"처럼 완료 여부부터 확인하세요. 완료했다고 "
                "답하면 record_task_progress(application_submitted=True)로 기록한 뒤 다음 "
                "턴부터 Step 2를 안내하세요."
            )
        return (
            "[진행 상태] Step 1의 하위 단계 [Step 1-1]~[Step 1-4] 안내가 모두 끝났고 신청ᆞ접수도 "
            "확인됐습니다. 이번 턴에는 사용자가 안 물어봐도 "
            "\"다음은 Step 2(공사) 단계입니다\"처럼 존재를 먼저 짚어주고 get_construction_guide를 "
            "호출해 안내하세요 - 식품위생ᆞ사업자등록 등 창업 준비 트랙으로 곧장 건너뛰지 마세요."
        )

    return "[진행 상태] Step 1~2(건축 인허가ᆞ공사) 안내가 모두 끝났습니다."


# permit_phase_directive가 "이번 턴에 get_construction_guide를 먼저 호출하라"고
# 지시하는 시점(Step 1 다 끝남 + 아직 미호출)에, 그 지시를 어기고 실제로는
# Step 2 내용을 텍스트로만 설명하고 넘어갔는지 판별하는 신호(guard.py의
# _construction_guide_gap_violations도 재사용). 이미 Step1 요약(예: "허가
# 신청 → 착공신고 → 공사 → 사용승인")에도 등장할 수 있는 "착공"ᆞ"감리"ᆞ
# "사용승인" 같은 낱말은 일부러 안 넣었다 - 실사용 세션(5db53ccb, 2026-07-29)
# 에서 실제로 관측된 건 "Step 2(공사 단계)"라는 표제를 달고 도구 호출 없이
# 안내를 시작한 경우였고, 이 표제 문구는 Step1 요약에는 등장하지 않아 오탐
# 위험이 적다.
_STEP2_KEYWORDS = ("Step 2", "공사 단계")


def _should_force_construction_guide(state: "AgentState") -> bool:
    """이번 턴에 get_construction_guide 호출을 tool_choice로 강제해도 되는지.

    permit_phase_directive는 "사용자가 안 물어봐도 먼저 짚어달라"고 매 턴
    요청하지만, 그 조건 그대로 tool_choice를 강제하면 이 구간(Step1 끝ᆞ
    Step2 미시작)에 사용자가 완전히 다른 주제(예: 간판 신고)를 물어봐도
    매번 공사 안내로 강제 전환되는 부작용이 생긴다 - 그래서 "AI가 방금
    Step 2를 먼저 제안했고, 사용자가 그 제안에 응답하는 턴"으로만 범위를
    좁혔다(_STEP2_KEYWORDS로 직전 AI 메시지가 실제 제안인지 확인). 정확히
    세션 5db53ccb(2026-07-29)에서 재현된 실패 패턴 - AI가 "다음은 Step
    2(공사 단계)입니다... 안내해 드릴까요?"라고 제안한 뒤 사용자가 "응
    알려줘"라고 답했는데 도구 호출 없이 텍스트로만 답한 사례 - 만 겨냥한다.

    2026-07-31: 컨텍스트 트리밍(A/B 실험, 효과 미입증으로 롤백)에 이은
    두 번째 시도. 이번엔 프롬프트 신뢰에 기대지 않고 API 레벨에서 구조적으로
    강제한다는 점이 다르다.

    2026-08-03: application_submitted 확인 게이트(permit_phase_directive)와
    반드시 같은 조건을 봐야 한다 - 안 그러면 Step1-4 안내를 마치는 문장에
    LLM이 "다음은 공사 단계입니다"처럼 _STEP2_KEYWORDS를 스스로 언급하는
    순간(신청ᆞ접수 확인 전이어도) 이 함수가 먼저 get_construction_guide를
    강제 호출해버려서, 소프트 게이트가 있으나 마나 해지는 실패가 실측으로
    확인됐다(스모크 테스트, 세션 f4081a6a).
    """
    disclosed_stage = state.get("disclosed_stage", {})
    if disclosed_stage.get(_STAGE_DOMAIN, 0) < 4:
        return False
    if disclosed_stage.get("construction_guide"):
        return False
    if not (state.get("task_progress") or {}).get("application_submitted"):
        return False
    messages = state.get("messages", [])
    if len(messages) < 2 or not isinstance(messages[-1], HumanMessage):
        return False
    if not isinstance(messages[-2], AIMessage):
        return False
    prior_text = extract_text(messages[-2].content)
    return any(kw in prior_text for kw in _STEP2_KEYWORDS)
