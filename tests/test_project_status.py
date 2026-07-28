"""compute_project_status()(src/project_status.py) 회귀 테스트.

이 함수는 재접속 요약 배너ᆞNext Action 위젯 등 여러 화면이 공유하는
"단일 진행 현황 소스"라 특히 중요하다(모듈 docstring의 "같은 데이터를
다르게 보여주는 문제" 원칙 참고) - 과거 실사용 세션(79430043)에서 건축
트랙이 안 끝난 채로 창업 준비 트랙이 먼저 진행되면 위젯이 계속 "Step1"
이라고만 말하는 버그가 실제로 있었고, 그걸 고치며 트랙을 분리했다. 이
파일은 그 분리가 여전히 올바르게 동작하는지, 그리고 progress%ᆞsummary
문구 조립이 경계 케이스에서 안 깨지는지 검증한다.
"""
from __future__ import annotations

from src.project_status import compute_project_status


def test_empty_state_is_not_started():
    result = compute_project_status({})
    assert result["progress"] == 0
    assert result["completed"] == []
    assert "아직 시작 전" in result["summary"]
    assert result["construction_track"]["current_step"] == "건축 유형 확인"
    assert result["startup_track"]["current_step"] == "창업 행정"


def test_construction_track_progress_does_not_leak_into_startup_track():
    """실사용 버그(세션 79430043) 회귀 - 건축 트랙이 안 끝난 채로 창업
    준비 트랙(식품위생 등)이 먼저 진행돼도, 두 트랙의 current_step은
    서로 독립적으로 계산돼야 한다."""
    state = {"food_result": "일반음식점영업"}
    result = compute_project_status(state)
    # 창업 트랙은 식품위생이 확인됐으니 다음 항목(소방시설)으로 넘어가 있어야 하고
    assert result["startup_track"]["current_step"] == "창업 행정"
    assert result["startup_track"]["next_action"] == "소방시설 확인"
    # 건축 트랙은 여전히 맨 처음(행위 유형 확인)이어야 한다 - "Step1"에
    # 멈춰 있던 옛날 버그처럼 창업 트랙 진행 상황에 영향받으면 안 된다.
    assert result["construction_track"]["current_step"] == "건축 유형 확인"


def test_step_completion_moves_current_step_forward_within_track():
    state = {"case_facts": {"act_type": "신축"}}
    result = compute_project_status(state)
    assert "건축 유형 확인" in result["completed"]
    assert result["construction_track"]["current_step"] == "건축 인허가"
    assert result["construction_track"]["next_action"] == "허가ᆞ신고 대상 판정"


def test_construction_track_fully_done_has_no_current_step():
    state = {
        "case_facts": {"act_type": "신축"},
        "permit_result": {"permit_type": "건축신고"},
        "task_progress": {
            "documents_prepared": True, "pre_diagnosis_checked": True,
            "construction_notice": True, "construction": True, "use_approval": True,
        },
    }
    result = compute_project_status(state)
    assert result["construction_track"]["current_step"] is None
    assert result["construction_track"]["next_action"] is None
    assert set(result["completed"]) >= {"건축 유형 확인", "건축 인허가", "공사"}
    # 건축 트랙만 끝났고 창업 트랙은 안 끝났으니 아직 축하 메시지는 아니다.
    assert "완료하셨어요" in result["summary"]
    assert "🎉" not in result["summary"]


def test_all_steps_done_reaches_full_progress_and_celebration_summary():
    state = {
        "case_facts": {"act_type": "신축"},
        "permit_result": {"permit_type": "건축신고"},
        "food_result": "일반음식점영업",
        "fire_result": [],
        "signage_result": "허가ᆞ신고 불필요",
        "task_progress": {
            "documents_prepared": True, "pre_diagnosis_checked": True,
            "construction_notice": True, "construction": True, "use_approval": True,
            "business_registration": True, "hygiene_education": True,
            "interior_equipment": True, "staff_registration": True, "opened": True,
        },
    }
    result = compute_project_status(state)
    assert result["progress"] == 100
    assert result["construction_track"]["current_step"] is None
    assert result["startup_track"]["current_step"] is None
    assert "🎉" in result["summary"]


def test_hires_staff_false_counts_staff_registration_as_done():
    """직원을 안 두는 1인 운영 사업자는 hires_staff=False만으로 '직원 등록'
    항목이 완료 처리돼야 한다(2026-07-28 수정 회귀) - staff_registration을
    따로 True로 안 줘도 된다. Step3까지 다 끝낸 상태에서 Step4의 인테리어만
    완료하고 hires_staff=False를 주면, '직원 등록'은 건너뛰고 다음 할 일이
    바로 '영업 시작'이어야 한다(그렇지 않으면 이 항목에서 진행률이 영원히
    100%를 못 채우는 옛 버그가 재발한 것)."""
    state = {
        "food_result": "해당없음", "fire_result": [], "signage_result": "허가ᆞ신고 불필요",
        "task_progress": {
            "business_registration": True, "hygiene_education": True,
            "interior_equipment": True, "hires_staff": False,
        },
    }
    result = compute_project_status(state)
    assert result["startup_track"]["current_step"] == "오픈 준비"
    assert result["startup_track"]["next_action"] == "영업 시작"


def test_progress_percentage_rounds_to_nearest_integer():
    # 총 15개 하위 항목 중 8개 완료 -> 8/15*100 = 53.33... -> 반올림 53
    state = {
        "case_facts": {"act_type": "신축"},
        "permit_result": {"permit_type": "건축신고"},
        "food_result": "일반음식점영업",
        "fire_result": [],
        "signage_result": "허가ᆞ신고 불필요",
        "task_progress": {
            "documents_prepared": True, "pre_diagnosis_checked": True,
            "construction_notice": True,
        },
    }
    result = compute_project_status(state)
    assert result["progress"] == 53
