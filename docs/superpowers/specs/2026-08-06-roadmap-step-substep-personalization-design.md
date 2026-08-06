# 챗봇 단계→하위단계 안내 일반화 + 개인화 로드맵

## Context

로드맵 체크리스트(`src/static/index.html`의 `buildChecklists()`, 2026-08-06 커밋
`0e545a3`ᆞ`dd44f34`ᆞ`e9e8099`)는 Step0~4 각 항목에 주체ᆞ법적근거(actorNote)ᆞ
하위 실무 행위(children)ᆞ잠금(locked)ᆞ해당없음(notApplicable)까지 갖춘 상세
구조를 갖게 됐다. 반면 챗봇 대화(`src/agent.py`→`src/roadmap.py`)는:

- Step1(인허가)ᆞStep2(공사)만 `permit_phase_directive`(if/elif 나열, 300줄+)로
  "하위단계 하나씩 확인→기록→다음 단계 공개"를 하드코딩된 긴 문자열로 안내.
- Step3~4(식품위생ᆞ소방ᆞ간판ᆞ사업자등록ᆞ위생교육ᆞ오픈준비)는 `_roadmap_status_summary`
  안의 손글씨 문단 하나로만 "병렬 진행, 시점 구분"을 뭉뚱그려 안내 — 개별 항목
  단위 안내ᆞ완료 확인ᆞ순서가 없다.

이번 요청: 챗봇도 체크리스트 수준의 상세한 단계→하위단계 설명을 하되, (1) 반복되는
하드코딩 문자열 없이 (2) 사용자 상태(task_progress/case_facts)를 계속 확인해서
(3) 그 사용자에게 실제로 맞는 순서ᆞ항목만 골라 안내하고 (4) 로드맵을 만들기 전에는
그 개인화 판단에 꼭 필요한 질문만 그때그때 하나씩 묻는 구조로 만든다.

**스코프**: 백엔드(`src/roadmap.py`ᆞ`src/project_status.py`ᆞ신규
`src/roadmap_steps.py`)와 챗봇 대화 흐름만 대상. `src/static/index.html`ᆞ
`intake.html`(프론트) 체크리스트ᆞ입력 폼은 이번 변경 대상이 아니다(이미 자체
개인화 로직을 갖고 있고, 별도 언어/런타임이라 공유 원치 않음 - 기존에도
`project_status.py`가 JS와 별도 구현을 유지해 온 것과 같은 관례).

## 기존 코드에서 발견한 드리프트 (선행 수정 필요)

통합 테이블을 만들려면 아래 두 불일치를 먼저 바로잡아야 한다 — 안 그러면 새
테이블이 이미 틀린 걸 그대로 베낀다:

1. `project_status.py`의 Step1 substep 목록에 **건축사사무소 선정 항목이
   없다**(사전검토ᆞ설계서류ᆞ신청접수 3개뿐). 채팅 쪽(`roadmap.py`)은 이미
   2026-08-04부터 Step1-1을 건축사선정으로 쓰고 있어 진행률 계산이 이 항목을
   놓치고 있었다.
2. `인테리어ᆞ장비 설치`가 `project_status.py`에는 Step4(오픈준비)로 잡혀
   있는데, UI는 2026-07-28에 이미 Step2(공사)로 옮겼다(시공의 일부라는 판단).
   두 소스가 그 이후로 갈라져 있었다.

새 테이블은 최신 의도(채팅/UI 쪽 기준: 건축사선정=Step1-1, 인테리어장비=Step2)를
따르고, `project_status.py`도 이 테이블을 그대로 참조해 자동으로 맞아떨어지게
한다.

## 데이터 구조 (`src/roadmap_steps.py`, 신규)

```python
@dataclass
class ResolvedSubstep:
    label: str
    done: bool
    actor_note: str | None = None      # 주체ᆞ법적근거 캡션(정적 텍스트, 체크리스트에서 이식)
    question: str | None = None        # 완료 확인 질문(없으면 "{label} 하셨어요?" 기본형)
    locked: bool = False
    lock_reason: str | None = None

@dataclass
class NeedsInput:
    question: str                      # 이 사용자 상황을 가르는 데 필요한 질문
    field: str                         # record_task_progress(field=...)로 기록할 키

@dataclass
class SubstepDef:
    key: str                                            # task_progress 필드명(순서ᆞ마커 식별용)
    resolve: Callable[[state], ResolvedSubstep | NeedsInput | None]
    # 반환값 3가지:
    #  - ResolvedSubstep: 이 사용자에게 해당하는 라벨ᆞ완료여부ᆞ주체 확정
    #  - NeedsInput: 아직 이 사용자 상황을 모름 - 답부터 받아야 다음 결정 가능
    #  - None: 이 사용자에겐 이 항목 자체가 해당없음(완전 스킵, 다음 substep으로)

@dataclass
class StepDef:
    index: int
    label: str
    hard_gate: bool             # True=Step1ᆞ2(순차, 마커 기반 하드 차단), False=Step0/3/4(소프트 유도)
    marker: str | None          # "Step 1" / "Step 2" (hard_gate일 때만)
    substeps: list[SubstepDef]

ROADMAP_STEPS: list[StepDef] = [...]  # Step0~4, 각 substep의 resolve() 구현
```

### 개인화 예시

- **Step1-1 건축사 선정** (`requires_licensed_architect`ᆞ`uses_agency` 기반,
  기존 `_pending_stage1_confirmation`의 분기 로직을 이 함수 안으로 이동):
  - `verdict is None` → `None`(정보 부족, 막지 않고 통과 — 기존 동작 그대로).
  - `verdict is True` → `ResolvedSubstep(label="건축사사무소 선정ᆞ설계 계약", ...,
    actor_note="건축법 제23조상 설계 의무 대상 - 건축주가 건축사사무소와
    계약합니다.", question="건축사사무소는 선정하셨어요?")`.
  - `verdict is False`이고 `uses_agency is None` → `NeedsInput(question="직접
    진행하실 건가요, 건축사사무소ᆞ행정사 등 대행업체 도움을 받으실 건가요?",
    field="uses_agency")`.
  - `verdict is False`이고 `uses_agency=True` → 대행업체 선정 확인
    `ResolvedSubstep`(기존 "그 업체는 선정하셨어요?" 문구 재사용).
  - `verdict is False`이고 `uses_agency=False` → `ResolvedSubstep(done=True)`
    (선정할 대상 없음, 즉시 통과 — 기존 동작 그대로).
- **Step2 착공신고ᆞ시공**: `task_progress.interior_only is True` → `None`(구조
  공사 없음, 해당없음).
- **Step2 substep 목록 자체가 `interior_equipment`(인테리어ᆞ장비 설치)를
  포함**하도록 새로 추가한다(위 드리프트 수정 항목) — `Step4`의 substep
  목록에서는 이 항목을 뺀다(중복 집계 방지). 소방시설은 별도 게이트 없이
  Step2-2("시공ᆞ공사감리ᆞ소방시설ᆞ인테리어ᆞ장비 설치") 안내 문구 안의
  `actor_note` 상세로만 얹는다(기존에도 별도 마커 없이 하나로 묶여 있던
  단위라 마커 개수를 안 늘림).
- **Step3 식품위생ᆞ위생교육**: `food_result`가 "해당없음"ᆞ"신고대상제외"면
  → `None`.
- **Step4 직원등록**: `hires_staff is False` → `ResolvedSubstep(done=True)`
  (4대보험 가입 대상 자체가 아님, 기존 project_status.py 규칙 그대로).
- **Step3 사업자등록**: `documents_prepared`(또는 `permitNotNeeded`)와
  `food_result` 확정 전엔 `locked=True, lock_reason="임대차계약서ᆞ영업신고
  확인 후 진행 가능"`.

기존 문자열 자산 재사용: 위 `question`ᆞ`actor_note` 값은 `tests/test_agent_guard.py`가
이미 부분 문자열로 검증 중인 기존 문구(`"건축사사무소는 선정하셨어요"` 등)를
그대로 이식한다 — 리팩터링이면서도 기존 테스트를 깨지 않는 것으로 회귀 위험을
낮춘다.

## 범용 엔진 (`src/roadmap.py`에 추가)

```python
def _next_pending_substep(step: StepDef, state) -> tuple[SubstepDef, ResolvedSubstep | NeedsInput] | None:
    """substep을 순서대로 훑어 resolve() 결과가 None(해당없음)이거나
    done=True인 건 건너뛰고, 첫 번째로 안 끝난(NeedsInput 또는
    ResolvedSubstep(done=False)) 항목을 반환. locked=True면 그 항목 앞에서
    멈추고(아직 등장할 차례 아님) 반환하지 않는다."""

def _disclosure_text(step: StepDef, substep_label_so_far: str,
                      result: NeedsInput | ResolvedSubstep, hard_gate: bool) -> str:
    """
    - NeedsInput: "[진행 상태] ... {question}" 하나만 묻는다(다른 것 안 물음).
    - ResolvedSubstep(done=False): "[진행 상태] [{marker}: {label}] 안내까지
      끝났습니다. {actor_note 있으면 한 줄} 이번 턴에는 다음으로 넘어가지 말고
      '{question}'처럼 완료 여부부터 확인하세요. 완료했다고 답하면
      record_task_progress({key}=True)로 기록한 뒤 다음 턴부터 안내하세요."
      hard_gate=True면 "[Step X-Y]로 넘어가지 마세요" 마커 캡 문장 추가(기존
      permit_phase_directive와 동일 강도 유지 - guard.py 하드 차단과 이 소프트
      유도가 같은 결론을 가리키게 함).
      hard_gate=False(Step3ᆞ4)면 마커 캡 대신 "사용자가 다른 항목을 직접
      물으면 그건 바로 답해도 됩니다"로 대체(기존 병렬 트랙 원칙 유지).
    """
```

**Step1-1 특수 분기는 여전히 코드(함수)로 남는다** — 이건 문제 되는 하드코딩이
아니라 진짜 3갈래 업무 로직이고, 시스템 전체에서 이 정도 분기가 필요한 곳은
여기 하나뿐이라 별도 조건부-DSL을 만드는 게 오히려 과설계(ponytail 판단, 이미
논의ᆞ합의됨). 대신 그 분기의 "결과를 문장으로 쓰는" 부분만 `_disclosure_text`
공용 템플릿을 타서, 6개 이상 존재하던 손글씨 블록이 하나로 준다.

## guard.py — 인터페이스 불변

`_max_allowed_stage`ᆞ`_max_allowed_construction_stage`ᆞ`_mentioned_stages`ᆞ
`_STAGE_DOMAIN`ᆞ`_STEP2_KEYWORDS`는 시그니처ᆞ반환값을 그대로 유지한다(내부
구현만 새 테이블 참조로 교체). guard.py의 하드 차단(정규식 기반 마커 탐지,
tool-call 기반 위반 탐지)은 이번 리팩터링과 무관 — 이 파일ᆞ관련 테스트는
수정하지 않는 것이 목표.

## project_status.py

`_step_substeps()`가 `ROADMAP_STEPS`의 각 `resolve(state)` 결과에서 `done`만
뽑아 기존 `(label, done)` 튜플 형태로 변환 — 이번에 발견한 드리프트 2건이
자동으로 해소된다(건축사선정 항목 생김, 인테리어장비가 Step2로 이동).
`compute_project_status`의 `_track_status`(공통 트랙 진행 계산)는 그대로 재사용.

## `_roadmap_status_summary`(Step3~4 안내) 변경

기존 손글씨 문단 중 "시점 구분"(공사 전ᆞ후 무엇을 하는지) 설명은 그대로
유지(이건 항목별 반복이 아니라 한 번만 말하는 맥락 정보라 정적 텍스트로 남기는
게 맞음). 그 아래에 `_next_pending_substep`으로 계산한 창업 준비 트랙의 다음
안 밝혀진 항목 하나를 소프트 유도 문구로 덧붙인다 — 항목이 늘어나도 문단을
새로 안 써도 됨.

## 영향받는 파일

- `src/roadmap_steps.py`(신규) — 데이터클래스 + `ROADMAP_STEPS`.
- `src/roadmap.py` — `_next_pending_substep`ᆞ`_disclosure_text` 추가,
  `permit_phase_directive`ᆞ`_roadmap_status_summary` 내부를 이걸로 교체.
  `_pending_stage1_confirmation`ᆞ`_pending_stage2_confirmation`은 유지하되
  내부에서 `ROADMAP_STEPS`의 `resolve()`를 호출하도록 재배선.
- `src/project_status.py` — `_step_substeps()`가 `ROADMAP_STEPS` 참조.
- `tests/test_roadmap_steps.py`(신규) — 각 substep `resolve()` 분기(개인화ᆞ
  NeedsInputᆞ해당없음) 경계값 테스트.
- `tests/test_agent_guard.py`ᆞ`tests/test_project_status.py`ᆞ
  `tests/test_roadmap_locks.py` — 최대한 무수정 목표, 문구 이식으로 깨지는
  곳만 최소 수정.

## 검증 방향

- `uv run pytest` 전체 통과(기존 200여 개 + 신규 테스트).
- 실제 LLM 스모크 테스트(CLAUDE.md §3, 유닛 테스트만으론 완료 보고 안 함):
  건축사 의무/대행/직접 케이스 각 1회, Step3~4 신규 유도 문구 1회, NeedsInput
  단일 질문만 나가는지 1회.
- 리팩터링 전/후로 `test_agent_guard.py`가 그대로(무수정) 통과하는지가 회귀
  여부의 1차 신호.

## 다음 우선순위(이번 스코프 밖, 기록만)

- `src/static/index.html`ᆞ`intake.html`의 프론트 체크리스트ᆞ입력 폼은 이번에
  안 건드림 — 백엔드 테이블과 표현이 갈릴 수 있는 채로 남는다(이미 있던
  관례). 프론트까지 같은 소스를 쓰게 하는 건 별도 논의 필요.
