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
    # 일반수선은 REQUIRED_FIELDS가 비어 있어 act_type만으로 "인허가불필요"가
    # 확정된다 - renovation_scope는 판정에 안 쓰이므로 기대하지 않는다
    # (처음엔 기대했다가 오답 처리됐는데, 모델이 옳았다 - 2026-08-02).
    ("permit", "벽지랑 바닥재만 새로 바꾸는 단순 수선이에요.", {"act_type": "일반수선"}),
    # 대수선은 renovation_scope를 실제로 쓴다(True면 규모ᆞ층수까지 봐야 하고,
    # False면 그 자리에서 "인허가불필요"로 확정) - 그래서 여기선 기대한다.
    ("permit", "내력벽은 안 건드리고 창호만 교체하는 공사입니다.",
     {"act_type": "대수선", "renovation_scope": False}),
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
    # sells_ready_made_only는 표현을 바꿔가며 4건 둔다. 2026-08-02 측정에서 이
    # 필드만 Gemini가 못 뽑았는데(Gemma4는 뽑음), 1건짜리 표본으론 그 모델의
    # 일반적 약점인지 이 문장 특유의 문제인지 갈리지 않는다. 없으면 판정이
    # None에 머물러 식품 영업 종류가 영영 안 나오는 필드라 중요도가 높다.
    ("food", "조리는 안 하고 완제품 포장식품만 판매할 거예요.",
     {"serves_food": True, "sells_ready_made_only": True}),
    ("food", "만들지는 않고 이미 완성된 제품만 그대로 팔 예정입니다.",
     {"serves_food": True, "sells_ready_made_only": True}),
    ("food", "포장된 샌드위치를 떼다가 그대로 판매만 합니다.",
     {"serves_food": True, "sells_ready_made_only": True}),
    ("food", "즉석에서 조리해서 팔 거예요, 완제품 판매는 아닙니다.",
     {"serves_food": True, "sells_ready_made_only": False}),
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
    # 간판은 sign_type마다 판정에 쓰는 필드가 다르다(signage.py의
    # REQUIRED_FIELDS_SIGNAGE 참고) - 기대 args도 그 종류에 실제로 쓰이는
    # 필드로만 적어야 한다. 처음엔 "돌출간판+floor", "현수막+length_m",
    # "지주이용간판+area_sqm"을 정답으로 뒀다가 전부 오답 처리됐는데, 확인해보니
    # 모델이 도구 설명의 적용 범위 표기(예: "(벽면이용간판) 설치하려는 층수")를
    # 정확히 따른 것이었고 정답지 쪽이 틀렸다(2026-08-02).
    #   벽면이용간판: length_m, floor, is_third_party_ad, area_sqm
    #   돌출간판    : height_m, area_sqm, is_medical_or_salon_sign
    #   지주이용간판: height_m
    #   현수막      : display_facility_area_sqm
    #   입간판      : sign_type만으로 판정 확정
    ("signage", "벽면에 붙이는 간판을 달려고 합니다.", {"sign_type": "벽면이용간판"}),
    ("signage", "돌출간판인데 지면에서 높이가 4미터입니다.",
     {"sign_type": "돌출간판", "height_m": 4}),
    ("signage", "입간판 하나 세워두려고요.", {"sign_type": "입간판"}),
    ("signage", "현수막을 걸 건데 게시시설 면적이 40제곱미터예요.",
     {"sign_type": "현수막", "display_facility_area_sqm": 40}),
    ("signage", "우리 가게 광고가 아니라 다른 업체 광고를 걸어주는 벽면간판이에요.",
     {"sign_type": "벽면이용간판", "is_third_party_ad": True}),
    ("signage", "높이 5미터짜리 지주이용간판을 세울 겁니다.",
     {"sign_type": "지주이용간판", "height_m": 5}),
]
