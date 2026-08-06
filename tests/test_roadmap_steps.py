"""ROADMAP_STEPS(src/roadmap_steps.py)의 resolve() 분기 경계값 테스트.

roadmap.py(챗봇 안내)ᆞproject_status.py(진행률 계산)가 공유하는 판정 테이블이라
여기서 분기를 잘못 두면 두 소비자가 동시에 틀린다. LLM을 호출하지 않는
순수 함수라 상태 dict만 흉내내면 결정론적으로 재현 가능하다."""
from __future__ import annotations

from src.agents.permit import PermitResult
from src.roadmap import _next_pending_substep
from src.roadmap_steps import ROADMAP_STEPS, NeedsInput, ResolvedSubstep

_OWNER_CONFIRMED = ROADMAP_STEPS[0].substeps[2]  # 건축주 여부 확인
_ARCHITECT = ROADMAP_STEPS[1].substeps[0]  # 건축사사무소 선정
_PRE_DIAGNOSIS = ROADMAP_STEPS[1].substeps[1]  # 사전 검토
_DOCUMENTS_PREPARED = ROADMAP_STEPS[1].substeps[2]  # 설계ᆞ서류 준비
_APPLICATION_SUBMITTED = ROADMAP_STEPS[1].substeps[3]  # 신청ᆞ접수
_CONSTRUCTION_NOTICE = ROADMAP_STEPS[2].substeps[0]
_CONSTRUCTION = ROADMAP_STEPS[2].substeps[1]
_FOOD = ROADMAP_STEPS[3].substeps[0]
_HYGIENE = ROADMAP_STEPS[3].substeps[3]
_BUSINESS_REGISTRATION = ROADMAP_STEPS[3].substeps[2]
_STAFF_REGISTRATION = ROADMAP_STEPS[4].substeps[0]


def _state(case_facts=None, task_progress=None, **extra) -> dict:
    return {"case_facts": case_facts or {}, "task_progress": task_progress or {}, **extra}


# === Step0 건축주 여부 확인 - case_facts.ownership 기반(2026-08-06 수정) ===
# 별도 task_progress 자기보고였다가, case_facts.ownership(record_case_facts로
# 폼ᆞ채팅 공용 수집)이 이미 같은 사실이라 사용자가 폼에서 답해도 체크박스가
# 절대 자동으로 안 채워져 "또 물어보는" 버그로 이어져 수정했다.

def test_owner_confirmed_unknown_when_ownership_not_recorded():
    result = _OWNER_CONFIRMED.resolve(_state())
    assert isinstance(result, ResolvedSubstep)
    assert result.done is False


def test_owner_confirmed_done_when_ownership_already_recorded():
    result = _OWNER_CONFIRMED.resolve(_state(case_facts={"ownership": "임차인"}))
    assert isinstance(result, ResolvedSubstep)
    assert result.done is True
    assert "임차인" in result.label


# === Step1 건축사사무소 선정 - 3갈래 + None ===

def test_architect_mandatory_not_selected():
    result = _ARCHITECT.resolve(_state(case_facts={"act_type": "신축"}))
    assert isinstance(result, ResolvedSubstep)
    assert result.done is False
    assert result.question == "건축사사무소는 선정하셨어요?"
    assert "제23조" in result.actor_note
    assert result.content_guide is not None


def test_architect_mandatory_already_selected():
    result = _ARCHITECT.resolve(
        _state(case_facts={"act_type": "신축"}, task_progress={"architect_selected": True})
    )
    assert isinstance(result, ResolvedSubstep)
    assert result.done is True


def test_architect_optional_needs_agency_choice():
    result = _ARCHITECT.resolve(_state(case_facts={"act_type": "일반수선"}))
    assert isinstance(result, NeedsInput)
    assert result.field == "uses_agency"


def test_architect_optional_agency_selected_pending():
    result = _ARCHITECT.resolve(
        _state(case_facts={"act_type": "일반수선"}, task_progress={"uses_agency": True})
    )
    assert isinstance(result, ResolvedSubstep)
    assert result.done is False
    assert result.question == "그 업체는 선정하셨어요?"
    assert result.content_guide is not None


def test_architect_optional_self_managed_has_no_content_guide():
    # 선정할 대상 자체가 없어 즉시 통과되는 분기 - 안내할 내용이 없다.
    result = _ARCHITECT.resolve(
        _state(case_facts={"act_type": "일반수선"}, task_progress={"uses_agency": False})
    )
    assert result.content_guide is None


# === Step1 사전 검토 - 건축사 의무 여부에 따라 안내 문구가 갈린다 ===

def test_pre_diagnosis_mentions_architect_handles_it_when_mandatory():
    result = _PRE_DIAGNOSIS.resolve(_state(case_facts={"act_type": "신축"}))
    assert "설계와 함께 사전검토도 진행" in result.content_guide


def test_pre_diagnosis_mentions_self_consultation_when_not_mandatory():
    result = _PRE_DIAGNOSIS.resolve(_state(case_facts={"act_type": "일반수선"}))
    assert "사전 상담" in result.content_guide


# === Step1 설계ᆞ서류 준비 - 대행 여부에 따라 예시 서식이 갈린다 ===

def test_documents_prepared_shows_agency_example_when_mandatory():
    result = _DOCUMENTS_PREPARED.resolve(_state(case_facts={"act_type": "신축"}))
    assert "대행업체(건축사사무소ᆞ행정사)가 작성ᆞ대행" in result.content_guide


def test_documents_prepared_shows_direct_example_when_self_managed():
    result = _DOCUMENTS_PREPARED.resolve(
        _state(case_facts={"act_type": "일반수선"}, task_progress={"uses_agency": False})
    )
    assert "본인이 직접 작성ᆞ준비" in result.content_guide


# === Step1 신청ᆞ접수 ===

def test_application_submitted_has_content_guide():
    result = _APPLICATION_SUBMITTED.resolve(_state())
    assert result.content_guide is not None
    assert "record_permit_synthesis" in result.content_guide


def test_architect_optional_agency_and_selected_done():
    result = _ARCHITECT.resolve(
        _state(
            case_facts={"act_type": "일반수선"},
            task_progress={"uses_agency": True, "architect_selected": True},
        )
    )
    assert isinstance(result, ResolvedSubstep)
    assert result.done is True


def test_architect_optional_self_managed_passes_immediately():
    result = _ARCHITECT.resolve(
        _state(case_facts={"act_type": "일반수선"}, task_progress={"uses_agency": False})
    )
    assert isinstance(result, ResolvedSubstep)
    assert result.done is True


def test_architect_verdict_none_is_not_applicable():
    result = _ARCHITECT.resolve(_state(case_facts={}))
    assert result is None


# === Step2 착공신고ᆞ시공 - interior_only 스킵 ===

def test_construction_notice_skipped_when_interior_only():
    result = _CONSTRUCTION_NOTICE.resolve(_state(task_progress={"interior_only": True}))
    assert result is None


def test_construction_skipped_when_interior_only():
    result = _CONSTRUCTION.resolve(_state(task_progress={"interior_only": True}))
    assert result is None


def test_construction_notice_resolved_when_not_interior_only():
    result = _CONSTRUCTION_NOTICE.resolve(_state(task_progress={"interior_only": False}))
    assert isinstance(result, ResolvedSubstep)
    assert result.done is False


# === Step3 식품위생ᆞ위생교육 - NA 판정이면 해당없음 ===

def test_food_registration_none_when_not_applicable():
    assert _FOOD.resolve(_state(food_result="해당없음")) is None
    assert _FOOD.resolve(_state(food_result="신고대상제외")) is None


def test_hygiene_education_none_when_food_not_applicable():
    assert _HYGIENE.resolve(_state(food_result="해당없음")) is None
    assert _HYGIENE.resolve(_state(food_result="신고대상제외")) is None


def test_food_registration_locked_until_use_approval():
    result = _FOOD.resolve(_state(food_result="일반음식점영업"))
    assert isinstance(result, ResolvedSubstep)
    assert result.locked is True

    result = _FOOD.resolve(
        _state(food_result="일반음식점영업", task_progress={"use_approval": True})
    )
    assert result.locked is False


# === Step3 사업자등록 lock 조합 ===

def test_business_registration_locked_without_documents_and_food():
    result = _BUSINESS_REGISTRATION.resolve(_state())
    assert result.locked is True


def test_business_registration_locked_with_only_documents():
    result = _BUSINESS_REGISTRATION.resolve(
        _state(task_progress={"documents_prepared": True})
    )
    assert result.locked is True


def test_business_registration_unlocked_when_documents_and_food_known():
    result = _BUSINESS_REGISTRATION.resolve(
        _state(food_result="일반음식점영업", task_progress={"documents_prepared": True})
    )
    assert result.locked is False


def test_business_registration_unlocked_when_permit_not_needed():
    result = _BUSINESS_REGISTRATION.resolve(
        _state(
            food_result="일반음식점영업",
            permit_result=PermitResult(permit_type="인허가불필요", procedures=[]),
        )
    )
    assert result.locked is False


# === Step4 직원등록 - hires_staff=False면 완료 처리 ===

def test_staff_registration_hires_staff_false_bypasses_lock():
    result = _STAFF_REGISTRATION.resolve(_state(task_progress={"hires_staff": False}))
    assert result.done is True
    assert result.locked is False


def test_staff_registration_locked_until_business_registration():
    result = _STAFF_REGISTRATION.resolve(_state())
    assert result.done is False
    assert result.locked is True


# === _next_pending_substep - locked 항목을 만나면 멈춘다 ===

def test_next_pending_substep_stops_at_locked_item():
    # 식품위생(첫 substep)이 아직 사용승인 전이라 잠겨 있으면, 뒤에 잠기지
    # 않은 항목(간판 등)이 있어도 건너뛰지 않고 그 자리에서 멈춘다. 식품위생ᆞ
    # 간판(key=food_result/signage_result)은 record_task_progress 대상이 아니라
    # 건너뛰므로(아래 참고), 실제로 멈추는 지점은 사업자등록의 lock이다 -
    # food_result는 알지만 documents_prepared/permitNotNeeded가 없어 여전히 잠김.
    state = _state(food_result="일반음식점영업")
    assert _next_pending_substep(ROADMAP_STEPS[3], state) is None


def test_next_pending_substep_skips_non_task_progress_keys():
    """key가 record_task_progress 필드가 아닌 substep(식품위생ᆞ간판 확인처럼
    판정 도메인이 별도로 채우는 값)은 미완료여도 유도 대상에서 제외하고
    다음 substep으로 건너뛴다 - 안 그러면 "완료했다고 답하면
    record_task_progress(food_result=True)로 기록하세요"처럼 존재하지 않는
    지시가 생긴다(실제 LLM 스모크 테스트 중 발견한 버그, 2026-08-06)."""
    state = _state(
        food_result="일반음식점영업",  # done=True지만 어차피 key 필터로도 스킵 대상
        task_progress={"documents_prepared": True},
    )
    result = _next_pending_substep(ROADMAP_STEPS[3], state)
    assert result is not None
    step, substep, resolved = result
    assert substep.key == "business_registration"
    assert resolved.done is False
    assert resolved.locked is False


def test_next_pending_substep_skips_done_and_not_applicable():
    state = _state(
        food_result="해당없음",  # 식품위생ᆞ위생교육 모두 None(해당없음)으로 스킵
        signage_result="허가ᆞ신고 불필요",  # 간판도 이미 판정됨(done)
        task_progress={"business_registration": True},  # 사업자등록도 done
    )
    result = _next_pending_substep(ROADMAP_STEPS[3], state)
    assert result is None
