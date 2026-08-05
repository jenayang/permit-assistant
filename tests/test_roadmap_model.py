"""src/roadmap_model.py(compute_roadmap_steps) 회귀 테스트.

기존엔 이 잠금 로직이 src/static/index.html에만 있어서 tests/
test_roadmap_locks.py가 JS 블록을 잘라내 Node subprocess로 검증했다
(tests/fixtures/roadmap_locks_check.js). roadmap_model.py가 그 로직의 단일
소스가 된 뒤로는(Phase 1) 같은 케이스를 순수 파이썬 단위테스트로 그대로
검증한다 - 2026-07-28에 추가된 규칙(식품위생ᆞ소방ᆞ간판ᆞ사업자등록ᆞ직원
등록ᆞ영업시작 잠금)을 이식했다.
"""
from __future__ import annotations

from src.roadmap_model import Step, Substep, compute_roadmap_steps, is_visually_locked


def _find(steps: list[Step], title: str, text_prefix: str) -> Substep:
    step = next(s for s in steps if s.title == title)
    return next(s for s in step.substeps if s.text == text_prefix or s.text.startswith(text_prefix))


def test_food_registration_locked_until_use_approval():
    state = {"food_result": "일반음식점", "fire_result": [], "task_progress": {"use_approval": False}}
    steps = compute_roadmap_steps(state)
    assert is_visually_locked(_find(steps, "사업자등록 및 위생교육", "식품위생"))

    state["task_progress"] = {"use_approval": True}
    steps = compute_roadmap_steps(state)
    assert not is_visually_locked(_find(steps, "사업자등록 및 위생교육", "식품위생"))


def test_fire_locked_until_construction():
    state = {"fire_result": ["소화기구"], "signage_result": "신고", "task_progress": {"construction": False}}
    steps = compute_roadmap_steps(state)
    assert is_visually_locked(_find(steps, "사용승인ᆞ완공검사", "소방시설"))

    state["task_progress"] = {"construction": True}
    steps = compute_roadmap_steps(state)
    assert not is_visually_locked(_find(steps, "사용승인ᆞ완공검사", "소방시설"))


def test_signage_locked_until_construction():
    state = {"fire_result": [], "signage_result": "신고", "task_progress": {"construction": False}}
    steps = compute_roadmap_steps(state)
    assert is_visually_locked(_find(steps, "간판신고 및 오픈 준비", "간판"))

    state["task_progress"] = {"construction": True}
    steps = compute_roadmap_steps(state)
    assert not is_visually_locked(_find(steps, "간판신고 및 오픈 준비", "간판"))


def test_business_registration_lock_combinations():
    cases = [
        (False, None, True),
        (True, None, True),
        (False, "일반음식점", True),
        (True, "일반음식점", False),
    ]
    for docs, food, expect_locked in cases:
        state = {"food_result": food, "fire_result": [], "task_progress": {"documents_prepared": docs}}
        steps = compute_roadmap_steps(state)
        locked = is_visually_locked(_find(steps, "사업자등록 및 위생교육", "사업자등록"))
        assert locked == expect_locked, f"documents_prepared={docs}, food={food}"


def test_business_registration_unlocked_when_permit_not_needed():
    # 인허가불필요 케이스는 documents_prepared가 영원히 안 채워지므로
    # permit_not_needed로도 잠금이 풀려야 한다(2026-08-04).
    state = {
        "food_result": "일반음식점", "fire_result": [],
        "task_progress": {"documents_prepared": False},
        "permit_result": {"permit_type": "인허가불필요"},
    }
    steps = compute_roadmap_steps(state)
    assert not is_visually_locked(_find(steps, "사업자등록 및 위생교육", "사업자등록"))


def test_staff_registration_locked_until_business_registration():
    state = {"task_progress": {"business_registration": False}}
    steps = compute_roadmap_steps(state)
    assert is_visually_locked(_find(steps, "간판신고 및 오픈 준비", "직원 등록"))

    state["task_progress"] = {"business_registration": True}
    steps = compute_roadmap_steps(state)
    assert not is_visually_locked(_find(steps, "간판신고 및 오픈 준비", "직원 등록"))


def test_staff_registration_hires_staff_false_bypasses_lock():
    state = {"task_progress": {"business_registration": False, "hires_staff": False}}
    steps = compute_roadmap_steps(state)
    assert not is_visually_locked(_find(steps, "간판신고 및 오픈 준비", "직원 등록"))


def test_opening_locked_until_all_step3_items_done():
    state = {
        "food_result": "일반음식점", "fire_result": [], "signage_result": "신고불필요",
        "task_progress": {"business_registration": True, "hygiene_education": False},
    }
    steps = compute_roadmap_steps(state)
    assert is_visually_locked(_find(steps, "간판신고 및 오픈 준비", "영업 시작"))

    state["task_progress"] = {"business_registration": True, "hygiene_education": True}
    steps = compute_roadmap_steps(state)
    assert not is_visually_locked(_find(steps, "간판신고 및 오픈 준비", "영업 시작"))


def test_architect_badge_reflects_requires_architect():
    # requires_licensed_architect()가 act_type만으로 확정되는 케이스(임계값
    # 계산 없이 True/False/None 세 갈래를 그대로 얻을 수 있는 조합)를 쓴다 -
    # "이전"은 예외 없이 True, "일반수선"은 예외 없이 False, act_type 없음은 None.
    def design_step(case_facts: dict) -> Step:
        state = {"case_facts": case_facts, "task_progress": {}}
        return next(s for s in compute_roadmap_steps(state) if s.title == "건축사사무소 선정 및 도면 작성")

    assert design_step({"act_type": "이전"}).architect is True
    assert design_step({"act_type": "일반수선"}).architect is False
    assert design_step({}).architect is None


def test_not_applicable_items_are_closed():
    steps = compute_roadmap_steps({"fire_result": [], "task_progress": {}})
    assert _find(steps, "사용승인ᆞ완공검사", "소방시설").not_applicable

    steps = compute_roadmap_steps({
        "fire_result": ["소화기구"], "signage_result": "허가ᆞ신고 불필요", "task_progress": {},
    })
    assert _find(steps, "간판신고 및 오픈 준비", "간판").not_applicable
    assert not _find(steps, "사용승인ᆞ완공검사", "소방시설").not_applicable

    steps = compute_roadmap_steps({"food_result": "신고대상제외", "fire_result": ["소화기구"], "task_progress": {}})
    assert _find(steps, "사업자등록 및 위생교육", "식품위생").not_applicable

    steps = compute_roadmap_steps({"fire_result": [], "task_progress": {"hires_staff": False}})
    assert _find(steps, "간판신고 및 오픈 준비", "직원 등록").not_applicable

    steps = compute_roadmap_steps({"food_result": "신고대상제외", "fire_result": [], "task_progress": {}})
    assert _find(steps, "사업자등록 및 위생교육", "위생교육").not_applicable

    steps = compute_roadmap_steps({"food_result": "일반음식점", "fire_result": [], "task_progress": {}})
    assert not _find(steps, "사업자등록 및 위생교육", "위생교육").not_applicable


def test_interior_only_closes_construction_substeps():
    steps = compute_roadmap_steps({"fire_result": [], "task_progress": {"interior_only": True}})
    assert _find(steps, "착공신고 및 공사", "착공신고 완료 확인").not_applicable
    assert _find(steps, "착공신고 및 공사", "시공 완료 확인").not_applicable

    steps = compute_roadmap_steps({"fire_result": ["소화기구"], "task_progress": {"interior_only": True}})
    assert not is_visually_locked(_find(steps, "사용승인ᆞ완공검사", "소방시설"))

    steps = compute_roadmap_steps({
        "fire_result": [], "signage_result": "신고", "task_progress": {"interior_only": True},
    })
    assert not is_visually_locked(_find(steps, "간판신고 및 오픈 준비", "간판"))


def test_parallel_badge_set_until_construction_step_done():
    steps = compute_roadmap_steps({"fire_result": [], "task_progress": {}})
    assert steps[7].parallel  # "사업자등록 및 위생교육"
    assert steps[8].parallel  # "간판신고 및 오픈 준비"

    done_construction_state = {
        "fire_result": [],
        "task_progress": {"construction_notice": True, "construction": True, "interior_equipment": True},
    }
    steps = compute_roadmap_steps(done_construction_state)
    assert not steps[7].parallel
    assert not steps[8].parallel


def test_countable_excludes_info_substeps():
    steps = compute_roadmap_steps({"fire_result": [], "task_progress": {}})
    design_step = next(s for s in steps if s.title == "건축사사무소 선정 및 도면 작성")
    assert all(not s.info for s in design_step.countable)
    assert any(s.info for s in design_step.substeps)
