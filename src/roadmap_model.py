"""로드맵 Step1~9의 단일 계산 소스.

지금까지 이 계산은 세 곳에 따로 있었다 - src/static/index.html의
renderRoadmap()(실제 UI 렌더링, JS 객체 리터럴), src/project_status.py(진행률ᆞ
현재 단계 계산용 5그룹 축약), src/roadmap.py(대화 안내 문구용 잠금 bool). 이
모듈은 그중 index.html의 steps=[...] 리터럴 생성 블록을 그대로(새 판단 로직
추가 없이) 포팅한 것을 유일한 소스로 삼는다 - project_status.py는 이 결과를
5그룹으로 축약만 하고, roadmap.py는 여기 정의된 잠금 predicate를 import해서
쓴다(Phase 2ᆞ3), index.html은 이 모듈이 노출하는 API 응답을 그대로 렌더링만
하게 된다(Phase 5).

판정 로직(classify_case/classify_food_business/classify_fire_safety/
classify_signage, permit_thresholds.yaml)은 여기서 한 글자도 다시 계산하지
않는다 - case_facts/permit_result/food_result/fire_result/signage_result처럼
규칙 엔진이 이미 계산해 상태에 실어둔 값을 Step 구조로 배치만 한다. 루트
CLAUDE.md §1 "판정은 규칙 엔진, LLM 대체 금지" 원칙과 무관한 순수 배관 계층.

체크박스 3종:
- field 지정: 사용자가 직접 체크(task_progress에 저장, 진행률 집계 O).
- info+check_key 지정: 참고용 목록(서류ᆞ사전진단 항목 등). done은 이 함수가
  항상 False로 반환한다 - 실제 체크 여부는 법적ᆞ판정 의미가 없는 개인 메모라
  서버로 옮기지 않고 클라이언트 localStorage(itemChecks)가 계속 소유한다
  (index.html이 이 값을 오버레이해서 그린다). 진행률 집계에서도 제외한다
  (Step.countable 참고).
- 위 둘 다 아님: 시스템이 case_facts/permit_result 등에서 자동 판정(사용자가
  못 뒤집음).

알려진 공백: "사업 예정지 주소 확인" substep이 참조하는 주소는 지금 어떤
tool도 case_facts에 쓰지 않는다(index.html도 client localStorage에 tool_call
인자를 재구성해서만 채움) - 이 함수는 case_facts.get("address")를 보되,
서버 상태에 아직 아무도 이 키를 채우지 않으므로 당분간 항상 미완료로 계산된다.
이번 리팩터링이 만든 새 공백이 아니라 기존 상태를 그대로 반영한 것뿐이다.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from src.agents.permit import requires_licensed_architect


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """permit_result가 운영에선 PermitResult(pydantic), 테스트에선 plain dict로
    들어올 수 있어 두 형태를 동일하게 다룬다."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass(frozen=True)
class Substep:
    text: str
    done: bool
    field: str | None = None
    check_key: str | None = None
    info: bool = False
    locked: bool = False
    lock_reason: str | None = None
    not_applicable: bool = False
    na_reason: str | None = None
    required: bool = False


@dataclass(frozen=True)
class Step:
    index: int  # 1-based (STEP 1 ~ STEP 9)
    title: str
    subtitle: str
    substeps: tuple[Substep, ...]
    architect: bool | None = None
    parallel: bool = False

    @property
    def countable(self) -> tuple[Substep, ...]:
        """info 항목은 완료 개념이 없으니 진행률 집계에서 제외한다."""
        return tuple(s for s in self.substeps if not s.info)

    @property
    def done(self) -> bool:
        items = self.countable
        return bool(items) and all(s.done for s in items)


def is_visually_locked(sub: Substep) -> bool:
    """상세 화면에서 실제로 잠긴 것으로 보이는지(index.html의
    renderStepDetail() 기준과 동일). notApplicable(✕)이 잠금보다 우선이고,
    field가 있는 substep은 이미 done되면 locked 값과 무관하게 더는 잠긴
    것으로 안 보인다(예: hires_staff=False로 우회된 직원 등록)."""
    if sub.info:
        return False
    if sub.not_applicable:
        return False
    if sub.field:
        return sub.locked and not sub.done
    return sub.locked


# --- 잠금 predicate -----------------------------------------------------
# roadmap.py(대화 페이싱 문구)와 이 모듈(Step 렌더링)이 같은 함수를 import해서
# "index.html의 lockReason과 반드시 같은 조건이어야 한다"는 기존의 주석상
# 약속을 코드로 강제한다.

def fire_signage_unlocked(task_progress: dict) -> bool:
    """소방시설ᆞ간판 설치ᆞ확인은 공사(시공) 완료 후 가능. 인테리어만 하는
    경우(interior_only)는 construction이 영원히 안 채워질 수 있어 예외."""
    return bool(task_progress.get("construction")) or bool(task_progress.get("interior_only"))


def food_report_unlocked(task_progress: dict) -> bool:
    """식품위생 영업신고 접수는 사용승인 완료 후 가능."""
    return bool(task_progress.get("use_approval"))


def business_registration_unlocked(task_progress: dict, food_result: str | None, permit_not_needed: bool) -> bool:
    """사업자등록(및 관련 서류ᆞ과세유형 확인)은 임대차계약서 등 서류 준비 +
    영업신고 종류 판정이 나온 뒤 가능. 인허가불필요 케이스는
    documents_prepared가 영원히 안 채워지므로 permit_not_needed로 대체."""
    return (bool(task_progress.get("documents_prepared")) or permit_not_needed) and food_result is not None


def staff_registration_unlocked(task_progress: dict) -> bool:
    """직원 등록(4대보험)은 사업자등록 완료 후 가능."""
    return bool(task_progress.get("business_registration"))


def opening_unlocked(task_progress: dict, food_result: str | None, signage_result: str | None) -> bool:
    """최종 점검ᆞ영업 시작은 식품위생ᆞ간판ᆞ사업자등록ᆞ위생교육 4개 모두
    끝나야 가능(소방시설은 사용승인 단계로 옮겨가 이 게이트에서 빠졌다)."""
    return (
        food_result is not None
        and signage_result is not None
        and bool(task_progress.get("business_registration"))
        and bool(task_progress.get("hygiene_education"))
    )


def compute_roadmap_steps(state: dict) -> list[Step]:
    """state(case_facts/permit_result/food_result/fire_result/signage_result/
    task_progress)에서 Step1~9 구조를 계산한다. LLM을 부르지 않는 순수
    결정론적 함수 - 이미 규칙 엔진이 계산해 상태에 실어둔 값을 배치만 한다."""
    case_facts = state.get("case_facts") or {}
    permit_result = state.get("permit_result")
    food_result = state.get("food_result")
    fire_result = state.get("fire_result")
    signage_result = state.get("signage_result")
    task_progress = state.get("task_progress") or {}

    act_type = case_facts.get("act_type")
    address = case_facts.get("address")
    permit_type = _attr(permit_result, "permit_type")
    classified = permit_type is not None
    permit_not_needed = permit_type == "인허가불필요"
    requires_architect = requires_licensed_architect(case_facts)

    diag = _attr(permit_result, "pre_diagnosis_items", []) or []
    docs = _attr(permit_result, "required_documents", []) or []
    interior_only = bool(task_progress.get("interior_only"))

    # Step 4 · 인허가 사전 검토
    pre_review_substeps: list[Substep] = [
        Substep(text=d, done=False, info=True, check_key=f"pre_diag:{i}:{d}")
        for i, d in enumerate(diag)
    ]
    pre_review_substeps.append(Substep(
        text="사전 검토 확인 완료", done=bool(task_progress.get("pre_diagnosis_checked")),
        field="pre_diagnosis_checked", not_applicable=permit_not_needed,
        na_reason="인허가 대상이 아니라 사전 검토가 필요 없습니다",
    ))

    # Step 3 · 건축사사무소 선정 및 도면 작성
    design_docs_substeps: list[Substep] = [
        Substep(
            text="건축사사무소 의뢰 및 계약 체결(직접 진행 시 진행 방식 확정)",
            done=bool(task_progress.get("architect_selected")), field="architect_selected",
            not_applicable=task_progress.get("uses_agency") is False,
            na_reason="직접 진행하기로 하셔서 건축사사무소 선정이 필요 없습니다",
        ),
        Substep(
            text="구조 검토(2층 이상ᆞ연면적 200㎡ 초과 등 해당 시 - 건축법 시행령 제32조)",
            done=False, info=True, check_key="design:structure_review",
        ),
    ]
    design_docs_substeps.extend(
        Substep(text=d, done=False, info=True, check_key=f"docs:{i}:{d}") for i, d in enumerate(docs)
    )
    design_docs_substeps.append(Substep(
        text="설계ᆞ서류 준비 완료 확인", done=bool(task_progress.get("documents_prepared")),
        field="documents_prepared", not_applicable=permit_not_needed,
        na_reason="이 행위는 건축 인허가(신고ᆞ허가) 대상이 아닙니다",
    ))

    # Step 5 · 허가ᆞ신고 신청ᆞ접수
    apply_step_title = f"{permit_type} 신청ᆞ접수" if (permit_type and not permit_not_needed) else "허가ᆞ신고 신청ᆞ접수"
    application_substeps = [
        Substep(text="신청서 작성 및 서명", done=False, info=True, check_key="apply:form",
                not_applicable=permit_not_needed, na_reason="이 행위는 건축 인허가(신고ᆞ허가) 대상이 아닙니다"),
        Substep(text="서류 취합(도면ᆞ대지사용권원 등 준비된 서류 정리)", done=False, info=True,
                check_key="apply:gather", not_applicable=permit_not_needed,
                na_reason="이 행위는 건축 인허가(신고ᆞ허가) 대상이 아닙니다"),
        Substep(
            text="관할 구청 건축과 방문 또는 온라인(세움터) 접수",
            done=bool(task_progress.get("application_submitted")), field="application_submitted",
            not_applicable=permit_not_needed, na_reason="이 행위는 건축 인허가(신고ᆞ허가) 대상이 아닙니다",
        ),
        Substep(text="처리 기간 확인(허가 15~20일ᆞ신고 3~5일ᆞ기재변경 3~7일 - 참고용)", done=False, info=True,
                check_key="apply:duration", not_applicable=permit_not_needed,
                na_reason="이 행위는 건축 인허가(신고ᆞ허가) 대상이 아닙니다"),
        Substep(text="보완 요청 시 서류 보완 대응(법령 근거 없음ᆞ실무 관행)", done=False, info=True,
                check_key="apply:supplement", not_applicable=permit_not_needed,
                na_reason="이 행위는 건축 인허가(신고ᆞ허가) 대상이 아닙니다"),
    ]

    step3_all_done = opening_unlocked(task_progress, food_result, signage_result)

    steps = [
        Step(
            index=1,
            title="주소 및 건축물 현황 확인",
            subtitle="건축물대장 조회ᆞ용도ᆞ연면적 확인",
            substeps=(
                Substep(
                    text=(f"사업 예정지 주소 확인: {address}" if address else "사업 예정지 주소 확인"),
                    done=bool(address), required=True,
                ),
                Substep(
                    text=(f"행위 유형 확인: {act_type}" if act_type else "행위 유형 확인(신축ᆞ증축ᆞ용도변경 등)"),
                    done=bool(act_type), required=True,
                ),
                Substep(
                    text="건축물대장ᆞ용도지역 조회",
                    done=case_facts.get("land_zone") is not None or case_facts.get("current_facility_group") is not None,
                    required=True, not_applicable=(act_type == "신축"),
                    na_reason="신축(빈 대지)은 기존 건축물대장이 없어 조회 대상이 아닙니다",
                ),
            ),
        ),
        Step(
            index=2,
            title="신고ᆞ허가 대상 판단",
            subtitle="허가ᆞ신고 대상 확인ᆞ건폐율ᆞ용적률 검토",
            substeps=(
                Substep(
                    text=(f"판정: {permit_type}" if permit_type else "허가ᆞ신고ᆞ기재변경 대상 판정"),
                    done=classified, required=True,
                ),
                Substep(
                    text=(
                        "건축사사무소 설계 의무 대상(건축법 제23조)" if requires_architect is True
                        else "건축사사무소 설계 의무 없음(직접 진행 가능)" if requires_architect is False
                        else "건축사사무소 설계 의무 대상 여부 확인(건축법 제23조)"
                    ),
                    done=requires_architect is not None, required=True,
                ),
                Substep(text="건폐율ᆞ용적률 법정 상한 이내 확인", done=False, info=True,
                        check_key="classify:coverage_ratio"),
            ),
        ),
        Step(
            index=3,
            title="건축사사무소 선정 및 도면 작성",
            subtitle="예상 2~4주 ᆞ 건축사 선임, 설계 도면 작성, 구조 검토",
            substeps=tuple(design_docs_substeps),
            architect=requires_architect,
        ),
        Step(
            index=4,
            title="인허가 사전 검토",
            subtitle="관할 구청 사전 협의ᆞ소방ᆞ위생 협의 여부 확인",
            substeps=tuple(pre_review_substeps),
        ),
        Step(
            index=5,
            title=apply_step_title,
            subtitle="예상 3~5주 ᆞ 허가 신청서, 도면, 건축물대장, 위임장 등",
            substeps=tuple(application_substeps),
        ),
        Step(
            index=6,
            title="착공신고 및 공사",
            subtitle="예상 4~8주 ᆞ 착공신고서, 공사감리, 소방시설, 인테리어ᆞ장비 설치",
            substeps=(
                Substep(
                    text="착공신고서ᆞ설계도서 제출, 공사감리자 지정 여부 확인(관할 구청 건축과ᆞ건축법)",
                    done=False, info=True, check_key="construction:notice_detail",
                    not_applicable=interior_only,
                    na_reason="구조 공사 없이 인테리어만 진행하는 경우라 착공신고 대상이 아닙니다",
                ),
                Substep(
                    text="착공신고 완료 확인", done=bool(task_progress.get("construction_notice")),
                    field="construction_notice", not_applicable=interior_only,
                    na_reason="구조 공사 없이 인테리어만 진행하는 경우라 착공신고 대상이 아닙니다",
                ),
                Substep(
                    text="소방시설공사업자를 통한 소방시설 설치, 완공검사증명서 발급ᆞ보관(소방시설법)",
                    done=False, info=True, check_key="construction:fire_detail",
                    not_applicable=interior_only,
                    na_reason="구조 공사 없이 인테리어만 진행하는 경우라 해당 없습니다",
                ),
                Substep(
                    text="시공 완료 확인", done=bool(task_progress.get("construction")), field="construction",
                    not_applicable=interior_only,
                    na_reason="구조 공사 없이 인테리어만 진행하는 경우라 해당 없습니다",
                ),
                Substep(text="인테리어ᆞ주방기기ᆞ결제단말기 설치(법령 근거 없음ᆞ실무 관행)",
                        done=bool(task_progress.get("interior_equipment")), field="interior_equipment"),
            ),
        ),
        Step(
            index=7,
            title="사용승인ᆞ완공검사",
            subtitle="예상 1~2주 ᆞ 사용승인 신청서, 소방완공검사, 정화조 확인",
            substeps=(
                Substep(
                    text=(f"소방시설: {'ᆞ'.join(fire_result)}" if fire_result else "소방시설"),
                    done=fire_result is not None,
                    not_applicable=isinstance(fire_result, list) and len(fire_result) == 0,
                    na_reason="연면적 기준상 설치 대상 소방시설이 없습니다",
                    locked=not fire_signage_unlocked(task_progress),
                    lock_reason="공사(시공) 완료 후 설치ᆞ확인 가능",
                ),
                Substep(text="사용승인 신청서 제출(관할 구청 건축과ᆞ건축법)", done=False, info=True,
                        check_key="use_approval:apply_detail"),
                Substep(
                    text="정화조 설치ᆞ처리대상인원 확인, 건축물대장 반영 확인(정화조: 환경부고시 / 건축물대장: 법령 근거 없음ᆞ실무 관행)",
                    done=False, info=True, check_key="use_approval:septic_detail",
                ),
                Substep(text="사용승인 완료 확인", done=bool(task_progress.get("use_approval")), field="use_approval"),
            ),
        ),
        Step(
            index=8,
            title="사업자등록 및 위생교육",
            subtitle="예상 1~2주 ᆞ 사업자등록증, 식품위생교육 이수, 영업신고",
            substeps=(
                Substep(
                    text=(f"식품위생: {food_result}" if food_result else "식품위생 영업신고"),
                    done=food_result is not None,
                    not_applicable=food_result in ("해당없음", "신고대상제외"),
                    na_reason="식품위생법상 별도 영업신고 대상이 아닙니다",
                    locked=not food_report_unlocked(task_progress),
                    lock_reason="사용승인 완료 후 신고 가능",
                ),
                Substep(
                    text="필요서류 준비(임대차계약서 사본, 영업신고증 사본 등 - 부가가치세법ᆞ실무 관행 혼합)",
                    done=False, info=True, check_key="bizreg:docs_detail",
                    locked=not business_registration_unlocked(task_progress, food_result, permit_not_needed),
                    lock_reason="임대차계약서ᆞ영업신고 확인 후 진행 가능",
                ),
                Substep(
                    text="간이ᆞ일반과세자 여부 확인(관할 세무서ᆞ세무사 상담 권장, 부가가치세법)",
                    done=False, info=True, check_key="bizreg:tax_type_detail",
                    locked=not business_registration_unlocked(task_progress, food_result, permit_not_needed),
                    lock_reason="임대차계약서ᆞ영업신고 확인 후 진행 가능",
                ),
                Substep(
                    text="사업자등록 신청 완료 확인(관할 세무서, 사업 개시일로부터 20일 이내ᆞ부가가치세법)",
                    done=bool(task_progress.get("business_registration")), field="business_registration",
                    locked=not business_registration_unlocked(task_progress, food_result, permit_not_needed),
                    lock_reason="임대차계약서ᆞ영업신고 확인 후 진행 가능",
                ),
                Substep(
                    text="위생교육 이수(사전교육 6시간, 원격교육 가능ᆞ식품위생법 시행령 제21조제8호)",
                    done=bool(task_progress.get("hygiene_education")), field="hygiene_education",
                    not_applicable=food_result in ("해당없음", "신고대상제외"),
                    na_reason="식품위생법상 영업자가 아니라 위생교육 대상이 아닙니다",
                ),
            ),
        ),
        Step(
            index=9,
            title="간판신고 및 오픈 준비",
            subtitle="예상 1주 ᆞ 옥외광고물 신고, 직원 등록, 최종 점검",
            substeps=(
                Substep(
                    text=(f"간판: {signage_result}" if signage_result else "간판ᆞ옥외광고물"),
                    done=signage_result is not None,
                    not_applicable=signage_result is not None and "불필요" in signage_result,
                    na_reason="허가ᆞ신고가 필요 없는 간판입니다",
                    locked=not fire_signage_unlocked(task_progress),
                    lock_reason="공사(시공) 완료 후 설치 가능",
                ),
                Substep(
                    text="서울시 옥외광고물 조례상 세부 규격 확인(설치 전, 조례 제9조의2)",
                    done=False, info=True, check_key="signage:ordinance_detail",
                    locked=not fire_signage_unlocked(task_progress),
                    lock_reason="공사(시공) 완료 후 설치 가능",
                ),
                Substep(
                    text="직원 등록(4대보험 가입 - 국민연금법ᆞ국민건강보험법ᆞ고용보험법ᆞ고용산재보험료징수법)",
                    done=bool(task_progress.get("staff_registration")) or task_progress.get("hires_staff") is False,
                    not_applicable=task_progress.get("hires_staff") is False,
                    na_reason="직원을 두지 않아 4대보험 가입 대상이 아닙니다",
                    field="staff_registration",
                    locked=not staff_registration_unlocked(task_progress),
                    lock_reason="사업자등록 완료 후 진행 가능",
                ),
                Substep(
                    text="최종 점검(사업자등록증ᆞ영업신고증 게시 등 - 법령 근거 없음ᆞ실무 관행)",
                    done=False, info=True, check_key="opening:final_check_detail",
                    locked=not step3_all_done,
                    lock_reason="식품위생ᆞ간판ᆞ사업자등록ᆞ위생교육 완료 후 가능",
                ),
                Substep(
                    text="영업 시작", done=bool(task_progress.get("opened")), field="opened",
                    locked=not step3_all_done,
                    lock_reason="식품위생ᆞ간판ᆞ사업자등록ᆞ위생교육 완료 후 가능",
                ),
            ),
        ),
    ]

    # 병렬 배지 - 건축 트랙의 마지막 단계(steps[5], "착공신고 및 공사")가 아직
    # 안 끝났으면, 창업 준비 트랙(steps[7]ᆞsteps[8])은 지금부터 병행 진행
    # 가능하다고 표시한다. project_status.py의 트랙 경계와 동일 기준.
    if not steps[5].done:
        steps[7] = replace(steps[7], parallel=True)
        steps[8] = replace(steps[8], parallel=True)

    return steps
