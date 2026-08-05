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
        "task_progress": {
            "construction_notice": True, "construction": True, "interior_equipment": True,
            "contractor_selected": True, "ventilation_installed": True,
        },
    }
    steps = compute_roadmap_steps(done_construction_state)
    assert not steps[7].parallel
    assert not steps[8].parallel


def test_countable_excludes_info_substeps():
    steps = compute_roadmap_steps({"fire_result": [], "task_progress": {}})
    # "사용승인ᆞ완공검사"는 정화조 확인 info substep을 permit_result와 무관하게
    # 항상 갖고 있다("건축사사무소 선정 및 도면 작성"은 구조 검토를 필드로
    # 전환하면서 permit_result가 없으면 info substep이 하나도 안 남을 수
    # 있어 이 테스트엔 안 맞다, 2026-08-05).
    use_approval_step = next(s for s in steps if s.title == "사용승인ᆞ완공검사")
    assert all(not s.info for s in use_approval_step.countable)
    assert any(s.info for s in use_approval_step.substeps)


# --- 2026-08-05 체크리스트 확장(사용자 GPT 피드백) - 신규 substep 25개 ------

def test_step1_lease_contract_not_applicable_for_owner():
    steps = compute_roadmap_steps({"case_facts": {"ownership": "소유자"}, "task_progress": {}})
    sub = _find(steps, "주소 및 건축물 현황 확인", "임차인인 경우 임대차계약서 준비")
    assert sub.not_applicable
    assert not sub.done

    steps = compute_roadmap_steps({
        "case_facts": {"ownership": "임차인"}, "task_progress": {"lease_contract_prepared": True},
    })
    sub = _find(steps, "주소 및 건축물 현황 확인", "임차인인 경우 임대차계약서 준비")
    assert not sub.not_applicable
    assert sub.done


def test_step1_area_and_floor_substeps_toggle_with_case_facts():
    steps = compute_roadmap_steps({"case_facts": {}, "task_progress": {}})
    assert not _find(steps, "주소 및 건축물 현황 확인", "연면적 확인").done
    assert not _find(steps, "주소 및 건축물 현황 확인", "층수 확인").done

    steps = compute_roadmap_steps({"case_facts": {"size_sqm": 85, "floors": 2}, "task_progress": {}})
    assert _find(steps, "주소 및 건축물 현황 확인", "연면적 확인").done
    assert _find(steps, "주소 및 건축물 현황 확인", "층수 확인").done


def test_step2_facility_group_change_not_applicable_unless_use_change():
    steps = compute_roadmap_steps({"case_facts": {"act_type": "신축"}, "task_progress": {}})
    assert _find(steps, "신고ᆞ허가 대상 판단", "시설군 변경 확인").not_applicable

    steps = compute_roadmap_steps({
        "case_facts": {
            "act_type": "용도변경", "current_facility_group": 8, "desired_facility_group": 7,
        },
        "task_progress": {},
    })
    sub = _find(steps, "신고ᆞ허가 대상 판단", "시설군 변경 확인")
    assert not sub.not_applicable
    assert sub.done


def test_step3_new_checklist_fields_toggle_with_task_progress():
    fields = {
        "현장 방문": "site_visit_done",
        "평면도 작성": "floor_plan_drafted",
        "구조 검토": "structure_review_done",
        "소방 검토": "fire_review_done",
        "필요 시 관할 구청ᆞ소방서 등 협의": "consultation_done",
        "건축물대장 사본 준비": "ledger_copy_ready",
        "현장사진 촬영": "site_photos_taken",
        "기존 도면 준비": "existing_drawings_ready",
    }
    empty_steps = compute_roadmap_steps({"task_progress": {}})
    for text_prefix in fields:
        assert not _find(empty_steps, "건축사사무소 선정 및 도면 작성", text_prefix).done

    done_steps = compute_roadmap_steps({"task_progress": {v: True for v in fields.values()}})
    for text_prefix in fields:
        assert _find(done_steps, "건축사사무소 선정 및 도면 작성", text_prefix).done


def test_step5_poa_not_applicable_when_self_managed():
    steps = compute_roadmap_steps({
        "permit_result": {"permit_type": "신고"}, "task_progress": {"uses_agency": False},
    })
    assert _find(steps, "신고 신청ᆞ접수", "위임장 준비").not_applicable

    steps = compute_roadmap_steps({
        "permit_result": {"permit_type": "신고"},
        "task_progress": {"uses_agency": True, "poa_prepared": True},
    })
    sub = _find(steps, "신고 신청ᆞ접수", "위임장 준비")
    assert not sub.not_applicable
    assert sub.done


def test_step6_contractor_and_ventilation_toggle():
    steps = compute_roadmap_steps({"fire_result": [], "task_progress": {}})
    assert not _find(steps, "착공신고 및 공사", "시공사 선정").done
    assert not _find(steps, "착공신고 및 공사", "환기시설 설치").done

    steps = compute_roadmap_steps({
        "fire_result": [], "task_progress": {"contractor_selected": True, "ventilation_installed": True},
    })
    assert _find(steps, "착공신고 및 공사", "시공사 선정").done
    assert _find(steps, "착공신고 및 공사", "환기시설 설치").done


def test_step7_fire_completion_inspection_locked_until_construction():
    steps = compute_roadmap_steps({"fire_result": ["소화기구"], "task_progress": {"construction": False}})
    assert is_visually_locked(_find(steps, "사용승인ᆞ완공검사", "소방 완공검사증명서 발급 확인"))

    steps = compute_roadmap_steps({"fire_result": ["소화기구"], "task_progress": {"construction": True}})
    assert not is_visually_locked(_find(steps, "사용승인ᆞ완공검사", "소방 완공검사증명서 발급 확인"))


def test_step7_building_ledger_created_locked_until_use_approval():
    steps = compute_roadmap_steps({"fire_result": [], "task_progress": {"use_approval": False}})
    assert is_visually_locked(_find(steps, "사용승인ᆞ완공검사", "건축물대장 생성 확인"))

    steps = compute_roadmap_steps({
        "fire_result": [], "task_progress": {"use_approval": True, "building_ledger_created": True},
    })
    sub = _find(steps, "사용승인ᆞ완공검사", "건축물대장 생성 확인")
    assert not is_visually_locked(sub)
    assert sub.done


def test_step9_opening_prep_fields_toggle():
    fields = ["pos_installed", "card_terminal_installed", "ingredients_ordered", "trial_run_done"]
    texts = ["POS 설치", "카드단말기 설치", "식자재 발주", "시범 운영"]
    empty_steps = compute_roadmap_steps({"fire_result": [], "task_progress": {}})
    for text in texts:
        assert not _find(empty_steps, "간판신고 및 오픈 준비", text).done

    done_steps = compute_roadmap_steps({"fire_result": [], "task_progress": {f: True for f in fields}})
    for text in texts:
        assert _find(done_steps, "간판신고 및 오픈 준비", text).done


def test_step8_detail_populated_on_core_items():
    steps = compute_roadmap_steps({"food_result": "일반음식점", "task_progress": {"use_approval": True}})
    step8 = next(s for s in steps if s.title == "사업자등록 및 위생교육")
    for sub in step8.substeps:
        assert sub.detail is not None
        assert {"why", "basis", "docs", "duration"} <= sub.detail.keys()
