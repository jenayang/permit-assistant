# Step1~9 체크리스트 확장 및 대화 안내 로직 재설계

## 배경

사용자(GPT 피드백)가 STEP1~9 각 단계의 "해야 할 일" 체크리스트를 상세화한
목록을 제공했다. 현재 `roadmap_model.py`의 Step1~9는 제목ᆞ순서는 이미
거의 일치하지만 하위 체크리스트 항목이 그보다 적다.

별개로, 챗봇이 실사용 세션(b846cc22)에서 옛 5그룹(Step0~4) 시절 설계인
`[Step 1-N]`/`[Step 2-N]` 텍스트 마커 기반 안내를 계속 사용해 사용자에게
혼란을 줬다. 이 마커는 `src/roadmap.py`가 정규식으로 파싱해 페이싱하고
`src/guard.py`가 개수를 세어 강제하는 별도 체계로, 새 Step1~9 체크리스트와
번호 체계가 겹쳐 혼란을 가중시킨다.

이번 작업은 두 가지를 함께 한다:
1. Step1~9 체크리스트 항목을 사용자가 준 목록에 맞게 확장한다.
2. 대화 안내 로직을 마커 기반에서, 매 턴 실제 체크리스트 상태(어떤 항목이
   끝났고 안 끝났는지)를 기반으로 안내하는 방식으로 재설계한다 — "다음
   단계만 계속 알려주는" 게 아니라 "현재 상태를 확인하고 체크리스트를
   한 단계씩 진행"하도록.

## 원칙 (CLAUDE.md 준수)

- 어떤 Step이 "현재 단계"인지, 어떤 substep이 done/locked인지는 계속
  `roadmap_model.compute_roadmap_steps()`(코드)가 결정론적으로 계산한다.
  LLM은 그 결과를 참고해 설명만 한다 — 이 원칙은 이번 작업으로 변하지
  않는다.
- 이번 체크리스트 확장은 법령 판정 규칙(허가/신고 여부 등)을 바꾸지
  않는다. 전부 사용자가 실제로 완료했는지 자기보고하는 실무 체크리스트
  항목이라 CLAUDE.md §1의 "판정 규칙 변경"에 해당하지 않는다.

## A. Step1~9 체크리스트 확장 (`src/roadmap_model.py`)

기존 항목은 그대로 두고(텍스트만 일부 보강), 아래 신규 항목을 추가한다.
"자동"은 이미 있는 `case_facts`/`permit_result` 값으로 시스템이 계산하는
substep(체크박스 없음), "필드"는 `task_progress`에 저장되는 사용자
체크 가능 substep(자동 완료 인식 + 수동 체크 둘 다 가능), "info"는
참고용 목록(로컬 체크만, 진행률 집계 제외)이다.

### STEP1. 주소 및 건축물 현황 확인
- (기존) 사업 예정지 주소 확인 — 자동
- (기존) 행위 유형 확인 — 자동
- (텍스트 보강) 건축물대장 조회ᆞ현재 용도ᆞ시설군 확인 — 자동
  (`current_facility_group`) — "현재 용도"ᆞ"시설군"은 코드상 같은 값이라
  하나로 통합
- (신규) 연면적 확인 — 자동 (`size_sqm`)
- (신규) 층수 확인 — 자동 (`floors`)
- (기존 항목에서 분리) 용도지역 조회 — 자동 (`land_zone`)
- (신규) 건축주 여부 확인(소유ᆞ임차) — 자동 (`ownership`, 이미 존재하는
  필드)
- (신규) 임차인인 경우 임대차계약서 준비 — 필드 `lease_contract_prepared`
  (소유자면 not_applicable). STEP3에서도 같은 필드를 재사용한다(같은
  서류라 필드를 중복 생성하지 않음).

### STEP2. 신고ᆞ허가 대상 판단
- (기존) 판정 — 자동
- (신규) 시설군 변경 확인(용도변경 시) — 자동
  (`current_facility_group`+`desired_facility_group` 둘 다 있으면 완료,
  `act_type != "용도변경"`이면 not_applicable)
- (기존) 건축사사무소 설계 의무 대상 여부 — 자동
- (기존) 건폐율ᆞ용적률 법정 상한 이내 확인 — info

### STEP3. 건축사사무소 선정 및 도면 작성
- (기존) 건축사사무소 의뢰 및 계약 체결 — 필드 `architect_selected`
- (신규) 현장 방문 — 필드 `site_visit_done`
- (신규) 평면도 작성 — 필드 `floor_plan_drafted`
- (info→필드 전환) 구조 검토 — 필드 `structure_review_done`
- (신규) 소방 검토 — 필드 `fire_review_done`
- (신규) 필요 시 관할 구청ᆞ소방서 협의 — 필드 `consultation_done`
- (신규) 임대차계약서 지참 — 필드 `lease_contract_prepared`(STEP1과 동일
  필드 재사용)
- (신규) 건축물대장 사본 준비 — 필드 `ledger_copy_ready`
- (신규) 현장사진 촬영 — 필드 `site_photos_taken`
- (신규) 기존 도면 준비(있는 경우) — 필드 `existing_drawings_ready`
- (기존) 판정별 필요 서류 목록 — info(동적, `required_documents`)
- (기존) 설계ᆞ서류 준비 완료 확인 — 필드 `documents_prepared`

### STEP4. 인허가 사전 검토
- 변경 없음(주차ᆞ정화조ᆞ소방ᆞ장애인ᆞ피난ᆞ위생 항목은 이미 판정 결과에서
  동적으로 채워짐 — `pre_diagnosis_items`)

### STEP5. 허가ᆞ신고 신청ᆞ접수
- (info→필드 전환) 신청서 작성 및 서명 — 필드 `application_form_prepared`
- (신규) 설계도서 제출 준비 — 필드 `design_docs_ready`
- (신규) 건축물대장 첨부 — 필드 `ledger_attached`
- (신규) 소유권 증빙 서류 준비 — 필드 `ownership_proof_ready`
- (신규) 위임장 준비(대행 신청 시) — 필드 `poa_prepared`
  (`uses_agency`가 False면 not_applicable)
- (신규) 수수료 납부 — 필드 `fee_paid`
- (기존) 관할 구청 방문ᆞ온라인 접수 — 필드 `application_submitted`
- (기존) 처리 기간 확인 — info
- (기존) 보완 요청 대응 — info

### STEP6. 착공신고 및 공사
- (기존) 착공신고서ᆞ설계도서 제출 안내 — info
- (기존) 착공신고 완료 확인 — 필드 `construction_notice`
- (신규) 시공사 선정 — 필드 `contractor_selected`
- (기존) 소방시설 설치 안내 — info
- (신규) 환기시설 설치 — 필드 `ventilation_installed`
- (기존) 시공 완료 확인 — 필드 `construction`
- (기존) 인테리어ᆞ주방기기ᆞ결제단말기 설치 — 필드 `interior_equipment`

### STEP7. 사용승인ᆞ완공검사
- (기존) 소방시설 판정 — 자동
- (신규) 소방 완공검사증명서 발급 확인 — 필드
  `fire_completion_inspection_done` (공사 완료 후 잠금 해제, 기존
  `fire_signage_unlocked`와 동일 조건)
- (info→필드 전환) 사용승인 신청서 제출 — 필드 `use_approval_applied`
- (신규) 현장검사 — 필드 `site_inspection_done`
- (기존) 정화조 확인 — info
- (기존) 사용승인 완료 확인 — 필드 `use_approval`
- (신규) 건축물대장 생성 확인 — 필드 `building_ledger_created`

### STEP8. 사업자등록 및 위생교육
- 신규 필드 추가 없음. "필요서류 준비" info 텍스트에 소방시설완비증명서
  언급을 추가하는 정도만 보강.
- `Substep`에 선택적 `detail`(why/basis/docs/duration 4개 문자열) 필드를
  추가하고, 이 Step의 핵심 5개 항목(식품위생 영업신고ᆞ필요서류ᆞ세금유형
  ᆞ사업자등록ᆞ위생교육)에 채운다. `index.html` 상세보기에서 `sub.detail`이
  있으면 클릭해 펼치는 아코디언을 추가한다.

### STEP9. 간판신고 및 오픈 준비
- (기존) 간판 판정 — 자동
- (기존) 옥외광고물 조례 세부규격 확인 — info
- (기존) 직원 등록(4대보험) — 필드 `staff_registration`
- (신규) POS 설치 — 필드 `pos_installed`
- (신규) 카드단말기 설치 — 필드 `card_terminal_installed`
- (신규) 식자재 발주 — 필드 `ingredients_ordered`
- (신규) 시범 운영 — 필드 `trial_run_done`
- (기존) 최종 점검 — info
- (기존) 영업 시작 — 필드 `opened`

## B. `record_task_progress` 스키마 확장 (`src/agents/roadmap_progress.py`)

신규 필드 25개(위 A절 "필드" 표시 전부, `lease_contract_prepared`는 1개로
카운트)를 `TASK_PROGRESS_FIELDS`와 함수 시그니처ᆞdocstring에 추가한다.
`/task-progress` API의 검증 목록도 이 단일 소스를 그대로 재사용하므로
자동으로 함께 확장된다 — 즉 새 필드는 챗봇이 대화로 확인해 자동 기록할
수도, 사용자가 상세보기 화면에서 직접 체크박스를 클릭해 수동으로 기록할
수도 있다(기존 `field` substep과 동일한 이중 경로, 추가 설계 불필요).

## C. 대화 안내 로직 재설계 (핵심)

`[Step 1-N]`/`[Step 2-N]` 텍스트 마커와 `guard.py`의 마커 개수 기반 사후
강제 검증을 제거한다. 대신 매 턴:

1. `src/roadmap.py`가 `compute_roadmap_steps(state)`를 호출해 **건축
   트랙의 현재 Step**(index 1~7 중 처음 `done=False`인 것)과, 잠금이
   풀렸다면 **창업 트랙의 현재 Step**(index 8~9 중 처음 `done=False`인
   것)을 찾는다.
2. 그 Step(들)의 `countable` substep 중 미완료ᆞ비잠금 항목을(텍스트 +
   대응 field/checkKey) SystemMessage로 주입한다.
3. Step3~7에 지금 `_ANSWER_PERMIT_STAGES`/`_ANSWER_CONSTRUCTION_STAGES`에
   흩어져 있는 풍부한 설명 지침(왜 이 단계를 하는지ᆞ본인/대행업체 구분ᆞ
   소요 기간 등)은 버리지 않고 Step 번호별 블록으로 재구성해서, 현재
   활성 Step에 해당하는 블록만 골라 붙인다(상시 전부 주입하던 것에서
   "현재 단계 것만"으로 바뀌어 토큰 사용량도 줄어든다).
4. 지시문 골자: "이 목록에서 사용자가 이미 말했거나 완료를 확인해준
   항목은 그 턴에 바로 `record_task_progress`로 기록하라. 아직 다루지
   않은 항목 중 다음 것부터 자연스럽게 안내하라. 이미 done인 항목은
   다시 설명하지 말고, 여러 항목을 한 번에 나열하지 말라."

`src/guard.py`에서 마커 카운팅에 의존하는 사후 재작성 강제 로직도 함께
제거한다. 이 시점부터 페이싱은 순수 프롬프트 지시로만 유도한다(강제
장치 없음 — 브레인스토밍에서 사용자가 명시적으로 선택한 트레이드오프).

## D. STEP8 상세 UI (`src/static/index.html`)

`Substep.detail`(why/basis/docs/duration)이 채워진 항목은 상세보기
화면에서 클릭 시 펼쳐지는 아코디언으로 표시한다. `detail`이 없는 항목은
기존과 동일하게(클릭 시 아무 것도 안 펼쳐짐) 렌더링한다.

## 테스트 계획

- `tests/test_roadmap_model.py`: 신규 substep 25개의 done/locked/
  not_applicable 계산 유닛 테스트(각 필드별 True/False/미기록 3케이스)
- `tests/test_project_status.py`: 5그룹 축약 로직이 substep 개수 변경에
  안 깨지는지 확인(파일 내 "위치 인덱스 참조" 주석이 명시한 회귀 지점)
- `src/roadmap.py`의 새 현재-단계 안내 로직에 대한 유닛 테스트(마커
  파싱 테스트는 제거ᆞ대체)
- `guard.py`에서 마커 카운팅 관련 테스트 정리
- LLM 스모크 테스트 1회(CLAUDE.md 규칙 3 — 실제 호출로 새 지시문이 의도한
  대로 동작하는지, 특히 "이미 done인 항목을 다시 설명하지 않는지" 확인)

## 범위 밖(이번에 안 함)

- STEP4 사전 검토 항목의 동적 생성 로직 변경 없음.
- `project_status.py`의 5그룹(Step0~4) 축약 구조 자체는 변경하지 않음
  (내부 substep 참조 인덱스만 유지되도록 주의).
- 판정 규칙(허가/신고/기재변경 임계값) 변경 없음.
