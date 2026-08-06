"""로드맵 Step0~4의 하위단계(substep) 판정 테이블 - 챗봇(src/roadmap.py)과
project_status.py가 공유하는 단일 소스(Phase 5, 2026-08-06).

이 파일이 생기기 전에는 챗봇(roadmap.py의 permit_phase_directive/
_roadmap_status_summary)이 Step1ᆞ2를 300줄+ if/elif 하드코딩 문자열로,
project_status.py는 (label, done) 튜플을 손으로 나열해서 각자 따로 관리했다 -
그 결과 두 소스가 갈라지는 드리프트가 실제로 있었다(project_status.py에
건축사선정 항목이 아예 없었고, 인테리어ᆞ장비 설치가 Step2/Step4로 서로
다르게 배치돼 있었다). 이 테이블 하나로 판정을 모으고 두 소비자가 같은
`resolve()` 결과만 다르게 표현하게 한다.

`resolve(state)`의 세 가지 반환값:
  - `ResolvedSubstep`: 이 사용자에게 해당하는 라벨ᆞ완료여부ᆞ주체 확정.
  - `NeedsInput`: 아직 이 사용자 상황을 모름 - 답부터 받아야 다음 결정 가능.
  - `None`: 이 사용자에겐 이 항목 자체가 해당없음(완전 스킵).

`state`는 AgentState의 부분집합(`case_facts`ᆞ`task_progress`ᆞ`food_result`ᆞ
`signage_result`ᆞ`permit_result`)만 있으면 되는 얕은 dict - 순수 함수라 어디서든
같은 결과를 낸다(§1 - 판정은 코드가, LLM은 설명만).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from src.agents.food_safety import NO_REPORT_RESULTS
from src.agents.permit import requires_licensed_architect


@dataclass
class ResolvedSubstep:
    label: str
    done: bool = False
    actor_note: str | None = None
    question: str | None = None
    locked: bool = False
    lock_reason: str | None = None
    # 이 항목의 실제 내용(법적 근거ᆞ서류 목록ᆞ작성주체 등)을 어떻게 설명해야
    # 하는지에 대한 상세 지시문 - Phase6(2026-08-06)에서 prompts.py의
    # _ANSWER_PERMIT_STAGES에 하드코딩돼 있던 Step 1-N별 정적 산문을 여기로
    # 옮겼다. 프론트(index.html)ᆞproject_status.py가 이미 label/actor_note를
    # 단일 소스로 보는 것과 같은 이유 - 순서ᆞ분기가 바뀌어도 챗봇 안내
    # 문구가 따로 놀지 않게. actor_note(한 줄 요약)보다 길고 구체적인 경우만
    # 채운다(짧은 항목은 None으로 비워도 됨).
    content_guide: str | None = None


@dataclass
class NeedsInput:
    question: str
    field: str


@dataclass
class SubstepDef:
    key: str
    resolve: Callable[[dict], "ResolvedSubstep | NeedsInput | None"]


@dataclass
class StepDef:
    index: int
    label: str
    hard_gate: bool
    marker: str | None
    substeps: list[SubstepDef] = field(default_factory=list)


def _permit_not_needed(state: dict) -> bool:
    permit_result = state.get("permit_result")
    if permit_result is None:
        return False
    return permit_result.permit_type == "인허가불필요"


# === Step 0: 건축 유형 확인 ===

def _resolve_act_type(state: dict) -> ResolvedSubstep:
    case_facts = state.get("case_facts") or {}
    return ResolvedSubstep(label="행위 유형 확인", done=bool(case_facts.get("act_type")))


def _resolve_permit_classified(state: dict) -> ResolvedSubstep:
    return ResolvedSubstep(
        label="허가ᆞ신고ᆞ기재변경 대상 판정",
        done=bool(state.get("permit_result")),
    )


def _resolve_owner_confirmed(state: dict) -> ResolvedSubstep:
    task_progress = state.get("task_progress") or {}
    return ResolvedSubstep(
        label="건축주 여부 확인(임차 시 계약서 준비)",
        done=bool(task_progress.get("owner_confirmed")),
        question="건축주이신가요, 임차인이신가요? 임차인이면 임대차계약서를 준비해두시면 좋아요.",
    )


# === Step 1: 건축 인허가 실무 ===
# 건축사사무소 선정(1-1)은 requires_licensed_architect 판정에 따라 3갈래로
# 갈리는 진짜 업무 로직이라 별도 조건부-DSL 없이 이 함수 하나로 남긴다(스펙
# 합의 - ponytail 판단, 이 정도 분기가 필요한 곳은 시스템 전체에서 여기 하나뿐).
def _resolve_architect_selection(state: dict) -> ResolvedSubstep | NeedsInput | None:
    case_facts = state.get("case_facts") or {}
    task_progress = state.get("task_progress") or {}
    verdict = requires_licensed_architect(case_facts)

    if verdict is None:
        return None  # 정보 부족 - 막지 않음(기존 동작)

    if verdict is True:
        return ResolvedSubstep(
            label="건축사사무소 선정ᆞ설계 계약",
            done=bool(task_progress.get("architect_selected")),
            actor_note="건축법 제23조상 설계 의무 대상 - 건축주가 건축사사무소와 계약합니다.",
            question="건축사사무소는 선정하셨어요?",
            content_guide=(
                "판정 결과(허가/신고/기재변경 등)ᆞ관할기관은 이미 Step0(행위 유형 확인) "
                "시점에 전달했으니 여기서 다시 설명하지 마세요 - 이 단계는 오직 건축사사무소를 "
                "선정해야 하는지ᆞ어떻게 하는지만 다룹니다. 설계 의무 대상이라 설계도서를 "
                "건축사사무소가 작성해야 한다는 점을 한 줄로 짚고(당신이 규모ᆞ공사 범위를 "
                "보고 재판단하거나 \"필요할 수도 있다\"처럼 흐리게 말하지 마세요) 선정을 "
                "안내하세요. 건축사사무소 선정 여부는 이후 사전검토ᆞ서류 준비 안내 방식"
                "(누가 진행하는지)에 계속 영향을 줍니다."
            ),
        )

    # verdict is False - 법적 의무는 아님(직접 가능)
    uses_agency = task_progress.get("uses_agency")
    if uses_agency is None:
        return NeedsInput(
            question="직접 진행하실 건가요, 건축사사무소ᆞ행정사 등 대행업체 도움을 받으실 건가요?",
            field="uses_agency",
        )
    if uses_agency:
        return ResolvedSubstep(
            label="대행업체(건축사사무소ᆞ행정사) 선정",
            done=bool(task_progress.get("architect_selected")),
            actor_note="설계 의무 대상은 아니지만 대행을 선택하셨습니다 - 건축주가 업체와 계약합니다.",
            question="그 업체는 선정하셨어요?",
            content_guide=(
                "판정 결과ᆞ관할기관은 이미 Step0에서 전달했으니 다시 설명하지 마세요. "
                "\"법적 의무는 아닙니다\"라고 명확히 밝힌 뒤, 대행업체(건축사사무소ᆞ행정사) "
                "도움을 받기로 하셨으니 그 업체 선정ᆞ계약 진행을 안내하세요. 건축사사무소 "
                "선정 여부는 이후 사전검토ᆞ서류 준비 안내 방식(누가 진행하는지)에 계속 "
                "영향을 줍니다."
            ),
        )
    # 직접 진행 - 선정할 대상이 없어 바로 통과
    return ResolvedSubstep(label="건축사사무소 선정ᆞ설계 계약", done=True)


def _resolve_pre_diagnosis(state: dict) -> ResolvedSubstep:
    case_facts = state.get("case_facts") or {}
    task_progress = state.get("task_progress") or {}
    verdict = requires_licensed_architect(case_facts)
    if verdict is True:
        agency_note = (
            "설계 의무 대상이면, 통상 그 건축사사무소가 설계와 함께 사전검토도 진행하니 "
            "사용자가 별도로 확인할 필요는 적다고 안내하세요."
        )
    else:
        agency_note = (
            "설계 의무 대상이 아니어도 주차대수ᆞ정화조 용량ᆞ소방시설 기준은 관할 구청 "
            "건축과나 소방서에 도면ᆞ현황을 지참해 사전 상담을 받는 게 실무적으로 "
            "안전하다고 안내하세요."
        )
    return ResolvedSubstep(
        label="사전 검토",
        done=bool(task_progress.get("pre_diagnosis_checked")),
        actor_note="위 항목(주차ᆞ정화조ᆞ피난 등)은 건축사사무소가 설계도서에 반영합니다 - 건축주는 결과를 확인만 하면 됩니다.",
        question="사전 검토 항목은 확인해 보셨어요?",
        content_guide=(
            "\"설계 전에 법적 기준(주차ᆞ소방 등)을 먼저 확인하는 단계입니다 - 여기서 "
            "문제가 나오면 설계를 수정해야 하므로 서류 준비 전에 끝내는 게 좋습니다\"처럼 "
            "왜 지금 이 단계를 하는지 한 줄로 먼저 짚은 뒤, 주차대수ᆞ정화조 용량ᆞ소방시설ᆞ"
            "장애인 편의시설ᆞ위생요구사항 중 실제 해당하는 것만 1~2줄+근거와 함께 안내하세요. "
            "이 항목들은 일반인이 스스로 판단하기 어려운 경우가 많으니, 항목 나열로 끝내지 "
            "말고 \"이걸 개인이 직접 확인할 수 있는지 vs 전문가ᆞ관공서 확인이 필요한지\"와 "
            "\"실제로 어떻게 확인하는지\"를 함께 안내하세요 - \"확인해 보셨어요?\"라고 완료 "
            f"여부만 묻고 끝내면 안 됩니다. {agency_note} 검색 결과에 구체적인 확인 절차"
            "(예: 사전 상담 창구, 필요 서류)가 있으면 그대로 전달하고, 없으면 \"관할 구청 "
            "건축과ᆞ소방서에 문의\"처럼 일반적인 확인 경로만 안내하세요 - 없는 절차를 "
            "지어내지 마세요. 안내한 항목들을 record_permit_synthesis(pre_diagnosis_items=[...])"
            "에 답변과 동일한 문구로 반드시 기록하세요 - 프론트엔드 진행 표시가 이 목록의 "
            "존재 여부로 단계 완료를 판단합니다."
        ),
    )


def _resolve_documents_prepared(state: dict) -> ResolvedSubstep:
    case_facts = state.get("case_facts") or {}
    task_progress = state.get("task_progress") or {}
    verdict = requires_licensed_architect(case_facts)
    uses_agency = verdict is True or bool(task_progress.get("uses_agency"))
    if uses_agency:
        example = (
            "예시 구조(대행업체 이용인 경우 - 문구는 상황에 맞게):\n"
            "**[건축주가 준비]**\n"
            "- 대지 소유ᆞ사용권원 서류(등기부등본ᆞ임대차계약서 등, 대행업체가 대신 "
            "발급받을 수 없는 본인 명의 서류)\n"
            "**[대행업체(건축사사무소ᆞ행정사)가 작성ᆞ대행]**\n"
            "- 신고ᆞ신청서(별지 서식) - 법적 명의자는 건축주지만 작성ᆞ제출은 대행업체가 "
            "진행, 건축주는 서명만\n"
            "- 평면도ᆞ배치도ᆞ내화ᆞ방화ᆞ피난ᆞ설비 도서 등 설계도서\n"
            "**[관공서 자체 확인]**\n"
            "- 용도변경 시 변경 전 평면도는 관공서가 건축물대장으로 직접 확인하므로 "
            "본인이 준비할 필요 없음"
        )
    else:
        example = (
            "예시 구조(직접 진행인 경우 - 문구는 상황에 맞게):\n"
            "**[본인이 직접 작성ᆞ준비]**\n"
            "- 신고ᆞ신청서(별지 서식)\n"
            "- 대지 소유ᆞ사용권원 서류(등기부등본ᆞ임대차계약서 등)\n"
            "- 평면도ᆞ배치도 등 설계도서(건축사 의무 대상이 아니므로 직접 작성 가능, "
            "부담되면 인테리어 업체ᆞ행정사에 맡기는 경우도 있음)"
        )
    return ResolvedSubstep(
        label="설계ᆞ서류 준비",
        done=bool(task_progress.get("documents_prepared")),
        question="설계도서ᆞ서류 준비는 다 되셨어요?",
        content_guide=(
            "\"설계도서 작성은 건축사사무소와 상담해서 준비하는 과정이라 보통 몇 주가 "
            "걸립니다\"처럼 시간이 걸리는 단계라는 감각을 먼저 준 뒤, \"어디에 무엇을 "
            "제출할 것인지 위한 서류\"를 실제 작성 주체 중심으로 안내하세요. 법적 명의자는 "
            "항상 건축주(신청인)이지만, 실제로 누가 작성하는지는 직접/대행 여부에 따라 "
            "다릅니다 - \"건축사사무소가 대행해준다\"는 법적 요건이 아니라 대행을 선택했을 "
            "때만 해당하는 실무 관행입니다(법조문은 \"신청서를 제출하는 자\"만 규정하고 "
            "작성 주체는 명시하지 않음 - 임의로 \"보통 대행해준다\"고 단정하지 마세요). "
            "필요 서류를 하위 항목으로 나열하되, 단순 나열이 아니라 각 서류 옆에 실제 작성 "
            f"주체를 함께 표시하세요. 다른 단계와 같은 마크다운 서식(굵게 **소제목**, 목록은 "
            f"- 불릿)을 그대로 쓰세요. {example}\n"
            "record_permit_synthesis(required_documents=[...])를 반드시 호출해 기록하세요."
        ),
    )


def _resolve_application_submitted(state: dict) -> ResolvedSubstep:
    task_progress = state.get("task_progress") or {}
    return ResolvedSubstep(
        label="신청ᆞ접수",
        done=bool(task_progress.get("application_submitted")),
        question="신청서 접수는 다 하셨어요?",
        content_guide=(
            "\"서류가 준비되면 실제로 접수해 허가/신고 수리를 받는 단계입니다\"처럼 짚은 뒤, "
            "관할 시ᆞ군ᆞ구청에 신청ᆞ신고서(별지 서식)와 준비한 서류를 제출(전자문서ᆞ온라인 "
            "접수 가능)하고, 관계 법령상 협의가 필요하면 협의기관 검토를 거쳐 허가/신고가 "
            "수리된다는 흐름을 한두 줄로 안내하세요(협의기관이 확인되면 "
            "record_permit_synthesis(related_agencies=[...])). 안내를 마치면서, 예상 소요 "
            "기간(허가 15~20일ᆞ신고 3~5일ᆞ기재변경 3~7일 등 통상적으로 알려진 수준)을 "
            "별도 헤더 없이 자연스러운 한 문장으로 덧붙이세요. search_regulations로 실제 "
            "처리기한을 확인했으면 그 값을 우선 쓰고, 못 찾았으면 위 통상치를 참고로만 "
            "안내하며 \"정확한 기간은 관할 구청에 확인하라\"고 덧붙이세요."
        ),
    )


# === Step 2: 공사 ===

def _resolve_construction_notice(state: dict) -> ResolvedSubstep | None:
    task_progress = state.get("task_progress") or {}
    if task_progress.get("interior_only"):
        return None
    return ResolvedSubstep(
        label="착공신고",
        done=bool(task_progress.get("construction_notice")),
        actor_note="건축주 명의로 신고합니다(건축법 제21조ᆞ시행규칙 제14조) - 서류 작성은 건축사사무소ᆞ시공사가 대행하는 경우가 많습니다.",
        question="착공신고는 하셨어요?",
    )


def _resolve_construction(state: dict) -> ResolvedSubstep | None:
    task_progress = state.get("task_progress") or {}
    if task_progress.get("interior_only"):
        return None
    return ResolvedSubstep(
        label="시공ᆞ공사감리ᆞ소방시설ᆞ인테리어ᆞ장비 설치",
        done=bool(task_progress.get("construction")),
        actor_note=(
            "시공사(건설산업기본법 등록업체)가 진행하고, 공사감리자(건축사)가 별도로 "
            "감리ᆞ확인합니다. 소방시설공사업법상 별도 면허를 가진 소방시설공사업자가 "
            "소방시설을 설치합니다(일반 시공사와 다름)."
        ),
        question="시공은 끝나셨어요?",
    )


def _resolve_interior_equipment(state: dict) -> ResolvedSubstep:
    task_progress = state.get("task_progress") or {}
    return ResolvedSubstep(
        label="인테리어ᆞ장비 설치",
        done=bool(task_progress.get("interior_equipment")),
    )


def _resolve_use_approval(state: dict) -> ResolvedSubstep:
    task_progress = state.get("task_progress") or {}
    return ResolvedSubstep(
        label="사용승인",
        done=bool(task_progress.get("use_approval")),
        actor_note="건축법 제22조ᆞ시행규칙 제16조 - 신청은 건축주 명의, 서류(공사감리완료보고서 등)는 감리자ᆞ시공사가 준비합니다.",
        question="사용승인은 받으셨어요?",
    )


# === Step 3: 창업 행정 (병렬) ===

def _resolve_food_registration(state: dict) -> ResolvedSubstep | None:
    food_result = state.get("food_result")
    if food_result in NO_REPORT_RESULTS:
        return None
    task_progress = state.get("task_progress") or {}
    return ResolvedSubstep(
        label=f"식품위생: {food_result}" if food_result else "식품위생 영업신고",
        done=food_result is not None,
        locked=not bool(task_progress.get("use_approval")),
        lock_reason="사용승인 완료 후 신고 가능",
        actor_note="영업자(사업주) 본인이 관할 보건소에 신고합니다.",
    )


def _resolve_signage(state: dict) -> ResolvedSubstep:
    signage_result = state.get("signage_result")
    task_progress = state.get("task_progress") or {}
    return ResolvedSubstep(
        label=f"간판: {signage_result}" if signage_result else "간판ᆞ옥외광고물 확인",
        done=signage_result is not None,
        locked=not (bool(task_progress.get("construction")) or bool(task_progress.get("interior_only"))),
        lock_reason="공사(시공) 완료 후 설치 가능",
        actor_note="영업자 본인 또는 간판 시공업체가 관할 구청에 신고ᆞ허가를 받습니다.",
    )


def _resolve_business_registration(state: dict) -> ResolvedSubstep:
    task_progress = state.get("task_progress") or {}
    documents_ready = bool(task_progress.get("documents_prepared")) or _permit_not_needed(state)
    food_known = state.get("food_result") is not None
    return ResolvedSubstep(
        label="사업자등록",
        done=bool(task_progress.get("business_registration")),
        locked=not (documents_ready and food_known),
        lock_reason="임대차계약서ᆞ영업신고 확인 후 진행 가능",
        actor_note="사업주 본인이 관할 세무서(또는 홈택스ᆞ민원24)에 신청합니다.",
    )


def _resolve_hygiene_education(state: dict) -> ResolvedSubstep | None:
    food_result = state.get("food_result")
    if food_result in NO_REPORT_RESULTS:
        return None
    task_progress = state.get("task_progress") or {}
    return ResolvedSubstep(
        label="위생교육 이수",
        done=bool(task_progress.get("hygiene_education")),
        actor_note="영업자(또는 종업원) 본인이 식품위생교육 지정기관에서 이수합니다.",
    )


# === Step 4: 오픈 준비 ===

def _resolve_staff_registration(state: dict) -> ResolvedSubstep:
    task_progress = state.get("task_progress") or {}
    if task_progress.get("hires_staff") is False:
        return ResolvedSubstep(label="직원 등록(4대보험 가입)", done=True)
    return ResolvedSubstep(
        label="직원 등록(4대보험 가입)",
        done=bool(task_progress.get("staff_registration")),
        locked=not bool(task_progress.get("business_registration")),
        lock_reason="사업자등록 완료 후 진행 가능",
        actor_note="사업주 본인이 근로복지공단ᆞ4대 사회보험공단에 신고합니다.",
    )


def _resolve_opened(state: dict) -> ResolvedSubstep:
    task_progress = state.get("task_progress") or {}
    return ResolvedSubstep(label="영업 시작", done=bool(task_progress.get("opened")))


ROADMAP_STEPS: list[StepDef] = [
    StepDef(
        index=0, label="건축 유형 확인", hard_gate=False, marker=None,
        substeps=[
            SubstepDef(key="act_type", resolve=_resolve_act_type),
            SubstepDef(key="permit_result", resolve=_resolve_permit_classified),
            SubstepDef(key="owner_confirmed", resolve=_resolve_owner_confirmed),
        ],
    ),
    StepDef(
        index=1, label="건축 인허가", hard_gate=True, marker="Step 1",
        substeps=[
            SubstepDef(key="architect_selected", resolve=_resolve_architect_selection),
            SubstepDef(key="pre_diagnosis_checked", resolve=_resolve_pre_diagnosis),
            SubstepDef(key="documents_prepared", resolve=_resolve_documents_prepared),
            SubstepDef(key="application_submitted", resolve=_resolve_application_submitted),
        ],
    ),
    StepDef(
        index=2, label="공사", hard_gate=True, marker="Step 2",
        substeps=[
            SubstepDef(key="construction_notice", resolve=_resolve_construction_notice),
            SubstepDef(key="construction", resolve=_resolve_construction),
            SubstepDef(key="interior_equipment", resolve=_resolve_interior_equipment),
            SubstepDef(key="use_approval", resolve=_resolve_use_approval),
        ],
    ),
    StepDef(
        index=3, label="창업 행정", hard_gate=False, marker=None,
        substeps=[
            SubstepDef(key="food_result", resolve=_resolve_food_registration),
            SubstepDef(key="signage_result", resolve=_resolve_signage),
            SubstepDef(key="business_registration", resolve=_resolve_business_registration),
            SubstepDef(key="hygiene_education", resolve=_resolve_hygiene_education),
        ],
    ),
    StepDef(
        index=4, label="오픈 준비", hard_gate=False, marker=None,
        substeps=[
            SubstepDef(key="staff_registration", resolve=_resolve_staff_registration),
            SubstepDef(key="opened", resolve=_resolve_opened),
        ],
    ),
]
