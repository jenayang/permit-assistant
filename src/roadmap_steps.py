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
    permit_result = state.get("permit_result") or {}
    return permit_result.get("permit_type") == "인허가불필요"


# === Step 0: 건축 유형 확인 ===

def _resolve_act_type(state: dict) -> ResolvedSubstep:
    case_facts = state.get("case_facts") or {}
    return ResolvedSubstep(label="행위 유형 확인", done=bool(case_facts.get("act_type")))


def _resolve_permit_classified(state: dict) -> ResolvedSubstep:
    return ResolvedSubstep(
        label="허가ᆞ신고ᆞ기재변경 대상 판정",
        done=bool(state.get("permit_result")),
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
        )
    # 직접 진행 - 선정할 대상이 없어 바로 통과
    return ResolvedSubstep(label="건축사사무소 선정ᆞ설계 계약", done=True)


def _resolve_pre_diagnosis(state: dict) -> ResolvedSubstep:
    task_progress = state.get("task_progress") or {}
    return ResolvedSubstep(
        label="사전 검토",
        done=bool(task_progress.get("pre_diagnosis_checked")),
        actor_note="위 항목(주차ᆞ정화조ᆞ피난 등)은 건축사사무소가 설계도서에 반영합니다 - 건축주는 결과를 확인만 하면 됩니다.",
        question="사전 검토 항목은 확인해 보셨어요?",
    )


def _resolve_documents_prepared(state: dict) -> ResolvedSubstep:
    task_progress = state.get("task_progress") or {}
    return ResolvedSubstep(
        label="설계ᆞ서류 준비",
        done=bool(task_progress.get("documents_prepared")),
        question="설계도서ᆞ서류 준비는 다 되셨어요?",
    )


def _resolve_application_submitted(state: dict) -> ResolvedSubstep:
    task_progress = state.get("task_progress") or {}
    return ResolvedSubstep(
        label="신청ᆞ접수",
        done=bool(task_progress.get("application_submitted")),
        question="신청서 접수는 다 하셨어요?",
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
