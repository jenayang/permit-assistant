"""src/static/index.html의 로드맵 Step3ᆞ4 선행조건 잠금(🔒) 로직 검증.

순수 JS 렌더 로직이라 파이썬으로 재작성하지 않고, tests/fixtures/
roadmap_locks_check.js가 index.html에서 실제 코드 블록을 그대로 잘라내
평가한 결과를 subprocess로 받아온다(conftest.py의 roadmap_lock_results
fixture, 세션당 1회만 node 실행). 이렇게 하면 index.html이 바뀌어도 이
테스트가 옛날 로직을 흉내내는 게 아니라 항상 최신 코드를 검증한다.

2026-07-28에 추가된 규칙(전부 이 파일에서 검증):
- 식품위생 영업신고: 사용승인 전까지 잠금
- 소방시설ᆞ간판: 공사(시공) 전까지 잠금
- 사업자등록: 서류 준비 + 영업신고 판정 전까지 잠금(기존)
- 직원 등록: 사업자등록 전까지 잠금(단, 직원 없음이면 예외)
- 영업 시작: Step3 5개 항목 전부 완료 전까지 잠금
"""
from __future__ import annotations


def test_food_registration_locked_until_use_approval(roadmap_lock_results):
    assert roadmap_lock_results["food: 사용승인 전엔 판정 나도 잠김"]
    assert roadmap_lock_results["food: 사용승인 후엔 잠금 해제"]


def test_fire_and_signage_locked_until_construction(roadmap_lock_results):
    assert roadmap_lock_results["fire: 공사 전엔 잠김"]
    assert roadmap_lock_results["signage: 공사 전엔 잠김"]
    assert roadmap_lock_results["fire: 공사 후엔 해제"]
    assert roadmap_lock_results["signage: 공사 후엔 해제"]


def test_business_registration_lock_combinations(roadmap_lock_results):
    assert roadmap_lock_results["business_registration: documents_prepared=false, food=null → locked=true"]
    assert roadmap_lock_results["business_registration: documents_prepared=true, food=null → locked=true"]
    assert roadmap_lock_results["business_registration: documents_prepared=false, food=일반음식점 → locked=true"]
    assert roadmap_lock_results["business_registration: documents_prepared=true, food=일반음식점 → locked=false"]


def test_staff_registration_locked_until_business_registration(roadmap_lock_results):
    assert roadmap_lock_results["staff: 사업자등록 전엔 잠김"]
    assert roadmap_lock_results["staff: 사업자등록 후엔 해제"]


def test_staff_registration_hires_staff_false_bypasses_lock(roadmap_lock_results):
    assert roadmap_lock_results["staff: hires_staff=false면 사업자등록 전이어도 안 잠김(done 우회)"]


def test_opening_locked_until_all_step3_items_done(roadmap_lock_results):
    assert roadmap_lock_results["opened: 5개 중 1개(위생교육)만 빠져도 잠김"]
    assert roadmap_lock_results["opened: 5개 전부 충족되면 해제"]
