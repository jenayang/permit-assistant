"""classify_case()(src/agents/permit.py) 경계값 회귀 테스트.

이 도구가 실제로 하는 일 중 가장 안전-critical한 부분이다 - 사용자에게
"허가 대상"이라고 말하느냐 "신고 대상"이라고 말하느냐를 가르는 순수 규칙
함수라, 여기서 경계값 하나만 잘못 바뀌어도(예: `size_sqm < 200`이 실수로
`<=`가 되는 식) 사용자가 잘못된 법적 판정을 그대로 믿게 된다. LLM도 외부
API도 안 쓰는 결정론적 함수인데도 지금까지 이 프로젝트에 자동 테스트가
하나도 없었다.

케이스 25개는 새로 설계한 게 아니라 docs/test_scenarios.md의 "Part A.
classify_case() 경계값 검증표"를 그대로 옮긴 것이다(2026-07-28) - 그 문서가
설계 스펙, 이 파일이 그 스펙의 자동 집행이다. 표가 바뀌면 이 파일도 같이
갱신할 것(문서 상단에도 명시돼 있음).
"""
from __future__ import annotations

import pytest

from src.agents.permit import (
    classify_case,
    facility_group_from_use_name,
    requires_licensed_architect,
)

# (설명, case_facts, 기대 결과) - docs/test_scenarios.md Part A 표의 순서ᆞ
# 번호를 그대로 따른다.
CASES = [
    ("01_증축_85_신고경계포함", {"act_type": "증축", "extension_size_sqm": 85}, "건축신고"),
    ("02_증축_85.01_경계바로초과", {"act_type": "증축", "extension_size_sqm": 85.01}, "건축허가"),
    ("03_개축_30_소규모", {"act_type": "개축", "extension_size_sqm": 30}, "건축신고"),
    ("04_재축_200_큰폭초과", {"act_type": "재축", "extension_size_sqm": 200}, "건축허가"),
    (
        "05_신축_도시지역_규모무관허가",
        {"act_type": "신축", "land_zone": "기타", "size_sqm": 50, "floors": 1},
        "건축허가",
    ),
    (
        "06_신축_관리지역_면적층수모두충족",
        {"act_type": "신축", "land_zone": "관리지역", "size_sqm": 199, "floors": 2},
        "건축신고",
    ),
    (
        "07_신축_관리지역_면적경계불충족",
        {"act_type": "신축", "land_zone": "관리지역", "size_sqm": 200, "floors": 2},
        "건축허가",
    ),
    (
        "08_신축_관리지역_층수경계불충족",
        {"act_type": "신축", "land_zone": "관리지역", "size_sqm": 199, "floors": 3},
        "건축허가",
    ),
    (
        "09_신축_농림지역_특례지역",
        {"act_type": "신축", "land_zone": "농림지역", "size_sqm": 100, "floors": 1},
        "건축신고",
    ),
    (
        "10_신축_자연환경보전지역_특례지역",
        {"act_type": "신축", "land_zone": "자연환경보전지역", "size_sqm": 100, "floors": 1},
        "건축신고",
    ),
    ("11_이전_필수필드없음_원칙대로허가", {"act_type": "이전"}, "건축허가"),
    ("12_대수선_시행령8개기준미해당", {"act_type": "대수선", "renovation_scope": False}, "인허가불필요"),
    (
        "13_대수선_신축과동일기준_신고",
        {"act_type": "대수선", "renovation_scope": True, "size_sqm": 199, "floors": 2},
        "건축신고",
    ),
    (
        "14_대수선_면적경계",
        {"act_type": "대수선", "renovation_scope": True, "size_sqm": 200, "floors": 2},
        "건축허가",
    ),
    (
        "15_대수선_층수경계",
        {"act_type": "대수선", "renovation_scope": True, "size_sqm": 100, "floors": 3},
        "건축허가",
    ),
    (
        "16_용도변경_동일시설군_기재변경",
        {"act_type": "용도변경", "current_facility_group": 7, "desired_facility_group": 7},
        "건축물대장기재변경",
    ),
    (
        "17_용도변경_하위군에서상위군_허가",
        {"act_type": "용도변경", "current_facility_group": 7, "desired_facility_group": 5},
        "용도변경허가",
    ),
    (
        "18_용도변경_상위군에서하위군_신고",
        {"act_type": "용도변경", "current_facility_group": 5, "desired_facility_group": 7},
        "용도변경신고",
    ),
    ("19_일반수선_필수필드없음", {"act_type": "일반수선"}, "인허가불필요"),
    (
        "20_가설건축물_목적자체가신고예외",
        {
            "act_type": "가설건축물", "temporary_purpose": "공사용",
            # 이 케이스는 신고_목적_예외에 걸려 기간ᆞ구조와 무관하게 신고로
            # 확정되지만, REQUIRED_FIELDS 검사(_missing_fields)는 세 필드가
            # 전부 None이 아니어야 통과시키므로 더미 값을 채운다.
            "temporary_duration_years": 1, "temporary_is_concrete": False,
        },
        "가설건축물신고",
    ),
    (
        "21_가설건축물_3년이하비철콘조",
        {
            "act_type": "가설건축물", "temporary_purpose": "창고",
            "temporary_duration_years": 3, "temporary_is_concrete": False,
        },
        "가설건축물신고",
    ),
    (
        "22_가설건축물_기간충족했지만철콘조라탈락",
        {
            "act_type": "가설건축물", "temporary_purpose": "창고",
            "temporary_duration_years": 3, "temporary_is_concrete": True,
        },
        "가설건축물허가",
    ),
    (
        "23_가설건축물_3년경계바로초과",
        {
            "act_type": "가설건축물", "temporary_purpose": "창고",
            "temporary_duration_years": 3.01, "temporary_is_concrete": False,
        },
        "가설건축물허가",
    ),
    ("24_신축_필수필드누락_미판정", {"act_type": "신축"}, None),
    ("25_최초상태_act_type도없음", {}, None),
]


@pytest.mark.parametrize("facts,expected", [(c[1], c[2]) for c in CASES], ids=[c[0] for c in CASES])
def test_classify_case_boundary_table(facts, expected):
    assert classify_case(facts) == expected


# --- requires_licensed_architect (건축법 제23조ᆞ제19조6항) -----------------
# True=건축사 필요(대행), False=개인 직접 가능, None=정보 부족.
ARCHITECT_CASES = [
    # 신축: 규모 무관 항상 필요
    ("신축_소규모여도_필요", {"act_type": "신축"}, True),
    # 증축ᆞ개축ᆞ재축: 85㎡ 경계
    ("증축_84_예외", {"act_type": "증축", "extension_size_sqm": 84}, False),
    ("증축_85_필요", {"act_type": "증축", "extension_size_sqm": 85}, True),
    ("재축_면적미상_미판정", {"act_type": "재축"}, None),
    # 대수선: renovation_scope + 200㎡/3층 경계
    ("대수선_비대수선_불필요", {"act_type": "대수선", "renovation_scope": False}, False),
    ("대수선_소규모_예외", {"act_type": "대수선", "renovation_scope": True, "size_sqm": 199, "floors": 2}, False),
    ("대수선_200_필요", {"act_type": "대수선", "renovation_scope": True, "size_sqm": 200, "floors": 2}, True),
    ("대수선_3층_필요", {"act_type": "대수선", "renovation_scope": True, "size_sqm": 100, "floors": 3}, True),
    ("대수선_규모미상_미판정", {"act_type": "대수선", "renovation_scope": True}, None),
    # 용도변경: 신고ᆞ기재변경은 불필요, 허가는 500㎡ 경계
    ("용변_신고_불필요", {"act_type": "용도변경", "current_facility_group": 7, "desired_facility_group": 8}, False),
    ("용변_기재변경_불필요", {"act_type": "용도변경", "current_facility_group": 7, "desired_facility_group": 7}, False),
    ("용변_허가_499_예외", {"act_type": "용도변경", "current_facility_group": 8, "desired_facility_group": 7, "size_sqm": 499}, False),
    ("용변_허가_500_필요", {"act_type": "용도변경", "current_facility_group": 8, "desired_facility_group": 7, "size_sqm": 500}, True),
    ("용변_허가_면적미상_미판정", {"act_type": "용도변경", "current_facility_group": 8, "desired_facility_group": 7}, None),
    # 이전: 건축허가라 필요
    ("이전_필요", {"act_type": "이전"}, True),
    # 제23조 비대상 행위
    ("일반수선_불필요", {"act_type": "일반수선"}, False),
    ("가설건축물_불필요", {"act_type": "가설건축물"}, False),
    # act_type 자체가 없으면 미판정
    ("act_type없음_미판정", {}, None),
]


@pytest.mark.parametrize(
    "facts,expected", [(c[1], c[2]) for c in ARCHITECT_CASES], ids=[c[0] for c in ARCHITECT_CASES]
)
def test_requires_licensed_architect(facts, expected):
    assert requires_licensed_architect(facts) is expected


# --- facility_group_from_use_name (건축물대장 주용도 → 시설군) ----------------
@pytest.mark.parametrize("use_name,expected", [
    ("제1종근린생활시설", 7),
    ("제2종근린생활시설", 7),  # group5의 "제2종 근린생활시설 중 다중생활시설"에 역포함되면 안 됨
    ("업무시설", 8),
    ("단독주택", 8),
    ("판매시설", 5),
    ("의료시설", 6),
    ("공장", 2),
    ("정보없음", None),
    ("", None),
])
def test_facility_group_from_use_name(use_name, expected):
    assert facility_group_from_use_name(use_name) == expected
