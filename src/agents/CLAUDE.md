# src/agents/ — 규칙 엔진ᆞRAG 도구 작업 규칙

> 루트 [`CLAUDE.md`](../../CLAUDE.md)의 규칙이 먼저 적용된다. 여기는 이 폴더
> (판정 규칙 엔진 + 도구)에만 해당하는 추가 컨벤션이다. 배경은
> [`MEMORY.md`](../../MEMORY.md) → domain-registry-pattern / rule-justification-pattern.

## 이 폴더의 파일 종류

- **판정 규칙 엔진**: `permit.py`(허가/신고/기재변경) · `food_safety.py`(식품위생)
  · `fire_safety.py`(소방) · `signage.py`(간판). 각각 `classify_*(facts: dict)`
  순수 함수를 가진다.
- **RAG 안내 도구**: `construction.py` · `business_registration.py` ·
  `hygiene_education.py` · `opening_checklist.py`. 법령 근거가 있으면
  `src/rag/regulation.py`의 `search_regulations`로 검색, 없으면(순수 실무
  체크리스트) 정적 텍스트. 전환 기준: `MEMORY.md` → static-vs-rag-content.
  (검색 도구 본체 `regulation.py`는 이 폴더가 아니라 `src/rag/`에 있다 —
  판정 규칙이 아니라 RAG 파이프라인 진입점이라 그쪽으로 옮겼다.)
- **조회 도구**: `site.py`(용도지역ᆞ건축물대장 등 외부 API).
- **공용**: `legal_data.py`(YAML 임계값 로더) · `roadmap_progress.py`
  (task_progress 기록).

## classify_fn 계약 (모든 판정 함수가 지켜야 함)

- 시그니처: `classify_*(facts: dict) -> <판정타입> | None`.
- **정보 부족이면 반드시 `None`** — 억지로 하나를 고르지 않는다. "판정 불가"와
  "확정된 해당없음"은 다르다(예: 완제품 판매+면적 미상=`None`, 완제품
  판매+300㎡ 미만=`"신고대상제외"`라는 확정값 — `MEMORY.md` →
  legal-interpretation-cases 참고).
- **결정론적**이어야 한다 — 같은 facts는 항상 같은 결과. LLM 호출 절대 금지.
- 법정 임계값(면적ᆞ층수 등 숫자)은 하드코딩하지 않고 `permit_thresholds.yaml`
  + `legal_data.get_thresholds()`로 가져온다 — 법 개정 시 숫자만 고치면 되게.

## 새 판정 도메인을 추가할 때 (체크리스트)

1. `classify_<domain>(facts: dict) -> ... | None` 순수 함수 작성(위 계약 준수).
2. 판정 사유가 미묘하면(단순 참/거짓이 아니면) Rule Justification 패턴 적용 —
   "왜 이 판정인가"를 코드가 문장으로 만들어 반환/응답에 실을 것. LLM이
   즉흥적으로 이유를 설명하게 두지 말 것.
3. `record_<domain>_facts` 도구 작성 — 필드는 사용자가 실제로 답할 수 있는
   사실만(법령 용어를 그대로 필드명으로 쓰지 말 것, 예:
   `manufactures_or_cooks`는 되지만 `sells_ready_made_only`처럼 법령 요건과
   어긋나는 이름은 오판정으로 이어짐 — 실제 사례 있음).
4. `src/classify/classifier.py`의 `DOMAIN_CONFIGS`에 `_DomainConfig` 항목
   추가(facts_key, state_key, classify_fn, record_tool, keywords, domain) —
   그래프 배선은 건드릴 필요 없음.
5. `tests/test_<domain>_classify.py`에 경계값 회귀 테스트 작성(모든 분기ᆞ
   None 케이스 포함). 유닛 테스트는 LLM 호출 없이 결정론적으로 실행돼야 함.
6. Gold Set(`tests/fixtures/extraction_gold_set.py`)에 새 필드 추출 케이스
   추가 — 필드명이 도구 스키마와 일치하는지는 `test_gold_set_validity.py`가
   자동 확인한다.

## 판정 규칙(법령 해석) 변경

**혼자 판단하지 말 것.** 조문 원문을 `search_regulations`/`search_by_term`으로
직접 대조해서 근거를 제시하고, 사용자 결정을 받은 뒤 고칠 것 — 루트 CLAUDE.md
§1과 동일한 원칙. 과거 사례: `MEMORY.md` → legal-interpretation-cases.
