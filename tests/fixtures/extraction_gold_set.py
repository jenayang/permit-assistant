"""사실 추출 정확도 측정용 Gold Set.

`tool_choice` 강제로 record_* 호출 자체는 보장됐으므로(project_report.md 4-12),
다음 질문은 **부른 뒤에 올바른 사실을 뽑았는가**다. 이 파일은 그 측정의 정답지다.

각 항목은 (도메인, 발화, 기대 args)이고, **기대 args는 그 발화만 보고 확정할 수
있는 필드를 빠짐없이 적은 것**이다. 이 "빠짐없이"가 중요하다:
- 기대에 있는데 안 뽑히면 → Recall 손실(놓친 사실)
- 기대에 없는데 뽑히면   → Precision 손실(사용자가 말하지 않은 걸 지어냄)
두 번째가 이 프로젝트에선 특히 위험하다 - 지어낸 사실이 그대로 규칙 엔진에 들어가
잘못된 허가/신고 판정으로 직결되기 때문이다.

시설군 번호는 건축법 시행령 제14조제5항 기준(src/agents/permit.py FACILITY_GROUPS):
사무실=8(주거업무시설군), 카페ᆞ음식점ᆞ소매점=7(근린생활시설군), 공장=2(산업 등
시설군), 판매시설=5(영업시설군), 의료시설=6(교육 및 복지시설군).

주의: 여기 값을 고칠 때는 반드시 record_* 도구의 실제 시그니처와 대조할 것 -
필드 이름이 틀리면 측정이 조용히 왜곡된다.
"""
from __future__ import annotations

# (도메인, 발화, 기대 args)
GOLD_SET: list[tuple[str, str, dict]] = [
    # --- permit: act_type + 규모 ---
    ("permit", "성수동에 3층짜리 연면적 200제곱미터 건물을 신축하려고요.",
     {"act_type": "신축", "floors": 3, "size_sqm": 200}),
    ("permit", "기존 사무실을 카페로 용도변경하려고 합니다.",
     {"act_type": "용도변경", "current_facility_group": 8, "desired_facility_group": 7}),
    ("permit", "지금 있는 건물에 30제곱미터만 증축할 거예요.",
     {"act_type": "증축", "extension_size_sqm": 30}),
    ("permit", "내력벽을 철거하는 대수선 공사를 하려고 합니다.",
     {"act_type": "대수선", "renovation_scope": True}),
    ("permit", "벽지랑 바닥재만 새로 바꾸는 단순 수선이에요.",
     {"act_type": "일반수선", "renovation_scope": False}),
    ("permit", "공사용 가설건축물을 2년간 세워두려고 합니다.",
     {"act_type": "가설건축물", "temporary_purpose": "공사용", "temporary_duration_years": 2}),
    ("permit", "제가 건물 소유주이고, 1층 120제곱미터를 카페로 바꿀 겁니다.",
     {"act_type": "용도변경", "ownership": "소유자", "size_sqm": 120,
      "desired_facility_group": 7}),
    ("permit", "임차한 공장 건물을 판매시설로 용도변경하려고요.",
     {"act_type": "용도변경", "ownership": "임차인",
      "current_facility_group": 2, "desired_facility_group": 5}),
    ("permit", "관리지역에 연면적 150제곱미터 2층 건물을 새로 지으려 합니다.",
     {"act_type": "신축", "land_zone": "관리지역", "size_sqm": 150, "floors": 2}),
    ("permit", "소매점을 의원으로 바꾸려고 합니다.",
     {"act_type": "용도변경", "current_facility_group": 7, "desired_facility_group": 6}),

    # --- food ---
    ("food", "카페 창업하려고요.", {"serves_food": True}),
    ("food", "빵을 직접 구워서 파는 베이커리를 열 겁니다.",
     {"serves_food": True, "primarily_bakery": True}),
    ("food", "술도 같이 파는 식당을 하려고 합니다.",
     {"serves_food": True, "serves_alcohol": True}),
    ("food", "커피만 팔고 주류는 취급 안 합니다.",
     {"serves_food": True, "serves_alcohol": False}),
    ("food", "조리는 안 하고 완제품 포장식품만 판매할 거예요.",
     {"serves_food": True, "sells_ready_made_only": True}),
    ("food", "음식은 안 팔고 옷만 파는 매장입니다.", {"serves_food": False}),
    ("food", "빵도 직접 굽고 술은 안 팔아요.",
     {"serves_food": True, "primarily_bakery": True, "serves_alcohol": False}),

    # --- fire ---
    ("fire", "연면적 400제곱미터 매장인데 소방시설 뭐가 필요한가요?",
     {"size_sqm": 400}),
    ("fire", "백화점 안에 입점하는 매장이고 면적은 80제곱미터입니다.",
     {"size_sqm": 80, "is_large_store_tenant": True}),
    ("fire", "길가 단독 점포이고 연면적 250제곱미터예요. 소방 관련 문의드립니다.",
     {"size_sqm": 250, "is_large_store_tenant": False}),

    # --- signage ---
    ("signage", "벽면에 붙이는 간판을 달려고 합니다.", {"sign_type": "벽면이용간판"}),
    ("signage", "돌출간판을 3층에 설치하려고 해요.",
     {"sign_type": "돌출간판", "floor": 3}),
    ("signage", "입간판 하나 세워두려고요.", {"sign_type": "입간판"}),
    ("signage", "가로 5미터짜리 현수막을 걸 예정입니다.",
     {"sign_type": "현수막", "length_m": 5}),
    ("signage", "우리 가게 광고가 아니라 다른 업체 광고를 걸어주는 벽면간판이에요.",
     {"sign_type": "벽면이용간판", "is_third_party_ad": True}),
    ("signage", "면적 10제곱미터짜리 지주이용간판을 세울 겁니다.",
     {"sign_type": "지주이용간판", "area_sqm": 10}),
]
