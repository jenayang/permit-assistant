# 로드맵 역할ᆞ순서ᆞ분기 재설계 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 건축 인허가 판정을 Step0으로, 건축사사무소 선정을 새 Step1-1로 옮기고, 선정이 선택제일 때 직접/대행 분기ᆞ소방시설의 Step2 이동ᆞ구조공사 없는 인테리어 전용 케이스ᆞ마커 없이 새는 게이트 사각지대까지 한 번에 재배치한다.

**Architecture:** 기존 Step1(4단계)ᆞStep2(3단계) 페이싱 인프라(`[Step N-M]` 마커 정규식ᆞ`disclosed_stage` 카운터ᆞ`permit_phase_directive`ᆞ`guard_node`)는 그대로 두고, 각 번호가 가리키는 **내용**만 재배치한다. 게이트 판정 함수(`_pending_stage1_confirmation`/`_pending_stage2_confirmation`)를 단일 소스로 삼아 하드 캡(`_max_allowed_*`)ᆞ안내 문구(`permit_phase_directive`)ᆞ신규 툴콜 기반 가드 검증이 모두 같은 조건을 보게 한다.

**Tech Stack:** Python(LangGraph state machine), 순수 함수 + pytest, 프롬프트 텍스트(자연어), 바닐라 JS(index.html).

## Global Constraints

- 판정 로직(허가/신고 여부 등)은 LLM이 아니라 코드가 계산한다 - `classify_case`/`requires_licensed_architect` 등 기존 규칙 엔진 그대로 재사용, 새 규칙 엔진 추가 없음(이번 변경은 전부 "언제ᆞ누가ᆞ어떤 순서로 안내하는지"에 관한 것).
- 커밋은 각 Task 끝에서 1회씩, 사용자가 명시적으로 요청했을 때만(이 세션에서는 이미 "진행해" 승인을 받았으므로 Task별 커밋 진행).
- 모든 새 task_progress 필드는 `hires_staff`와 같은 패턴(완료 여부가 아니라 "선택ᆞ해당 여부"를 담는 bool)을 따른다.
- `uv run pytest` 전체 통과 + 실제 LLM 스모크 테스트 없이는 완료 보고하지 않는다(CLAUDE.md §3).

---

## Task 1: `roadmap_progress.py` - 새 task_progress 필드 3개 추가

**Files:**
- Modify: `src/agents/roadmap_progress.py`
- Test: `tests/test_roadmap_progress.py`

**Interfaces:**
- Produces: `TASK_PROGRESS_FIELDS`에 `"uses_agency"`, `"architect_selected"`, `"interior_only"` 추가. `record_task_progress` 도구 파라미터에 동일 이름 3개 추가(전부 `bool | None = None`).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_roadmap_progress.py` 맨 위 import 아래에 추가:

```python
from src.agents.roadmap_progress import TASK_PROGRESS_FIELDS


def test_new_role_fields_registered():
    assert "uses_agency" in TASK_PROGRESS_FIELDS
    assert "architect_selected" in TASK_PROGRESS_FIELDS
    assert "interior_only" in TASK_PROGRESS_FIELDS
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_roadmap_progress.py::test_new_role_fields_registered -v`
Expected: FAIL (`AssertionError`, "uses_agency" not in list)

- [ ] **Step 3: `src/agents/roadmap_progress.py` 수정**

`TASK_PROGRESS_FIELDS` 정의를 다음으로 교체:

```python
TASK_PROGRESS_FIELDS: list[str] = [
    "pre_diagnosis_checked", "documents_prepared", "application_submitted",
    "construction_notice", "construction", "use_approval",
    "business_registration", "hygiene_education",
    "interior_equipment", "staff_registration", "opened",
    "hires_staff", "uses_agency", "architect_selected", "interior_only",
]
```

`record_task_progress` 함수 시그니처에 3개 파라미터 추가(기존 `hires_staff: bool | None = None,` 바로 다음 줄):

```python
    hires_staff: bool | None = None,
    uses_agency: bool | None = None,
    architect_selected: bool | None = None,
    interior_only: bool | None = None,
) -> str:
```

docstring의 `Args:` 목록 끝(`hires_staff: ...` 다음)에 추가:

```
        uses_agency: 건축사사무소 설계가 법적 의무가 아닐 때, 대행업체(건축사사무소ᆞ행정사)
            도움을 받는지(True) 직접 진행하는지(False)
        architect_selected: 건축사사무소(또는 대행업체)를 실제로 선정 완료했는지
        interior_only: 구조 공사(신축ᆞ증축ᆞ대수선 등) 없이 인테리어만 진행하는지
```

docstring 본문 마지막 불릿(`hires_staff` 관련 설명) 다음에 추가:

```
    - **건축사사무소 선정이 의무가 아닐 때 "직접 하겠다/도움받겠다"를 명확히 밝히면
      uses_agency로 기록하세요** - 이후 사전검토ᆞ서류 준비 안내가 이 값에 따라
      갈립니다. 실제로 사무소ᆞ업체를 정했다고 밝히면 architect_selected=True로
      기록하세요.
    - **구조 공사 없이 인테리어만 진행한다고 명확히 밝히면 interior_only=True로
      기록하세요** - 착공신고ᆞ시공 항목 자체가 대상이 아니게 됩니다.
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_roadmap_progress.py -v`
Expected: PASS (전체)

- [ ] **Step 5: Commit**

```bash
git add src/agents/roadmap_progress.py tests/test_roadmap_progress.py
git commit -m "feat: 건축사 직접/대행ᆞ인테리어전용 판단용 task_progress 필드 3개 추가"
```

---

## Task 2: `roadmap.py` - Step0/1 게이트 재배치 + 툴콜 기반 검증 대비 헬퍼

**Files:**
- Modify: `src/roadmap.py`
- Test: `tests/test_agent_guard.py` (해당 부분만 이 Task에서 먼저 갱신 - 전체 스위트 정리는 Task 6)

**Interfaces:**
- Consumes: `src.agents.permit.requires_licensed_architect(case_facts: dict) -> bool | None`(기존).
- Produces: `_pending_stage1_confirmation(disclosed: int, task_progress: dict, case_facts: dict) -> int | None`(시그니처 변경 - case_facts 인자 추가), `_pending_stage2_confirmation(construction_disclosed: int, task_progress: dict) -> int | None`(interior_only 반영), `_max_allowed_stage(state) -> int`, `_max_allowed_construction_stage(state) -> int`, `permit_phase_directive(state) -> str | None`.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_agent_guard.py`에서 기존 `test_permit_phase_directive_none_before_classification` 함수 바로 다음에 추가(이 시점엔 아직 구현이 없어 실패해야 정상):

```python
def test_stage1_architect_gate_mandatory_blocks_until_selected():
    """건축사 의무 대상(True)이면 architect_selected 확인 전엔 [Step 1-2]로 못 간다."""
    state = _base_state()
    state["case_facts"] = {"_classified": True, "act_type": "신축"}
    state["disclosed_stage"] = {_STAGE_DOMAIN: 1}
    state["task_progress"] = {}
    directive = permit_phase_directive(state)
    assert "건축사사무소는 선정하셨어요" in directive

    state["task_progress"] = {"architect_selected": True}
    directive = permit_phase_directive(state)
    assert "[Step 1-2]까지만" in directive


def test_stage1_architect_gate_optional_asks_direct_or_agency():
    """건축사 선택제(False)면 먼저 직접/대행 여부를 물어야 한다."""
    state = _base_state()
    state["case_facts"] = {"_classified": True, "act_type": "일반수선"}
    state["disclosed_stage"] = {_STAGE_DOMAIN: 1}
    state["task_progress"] = {}
    directive = permit_phase_directive(state)
    assert "직접 진행하실 건가요" in directive


def test_stage1_architect_gate_optional_self_managed_passes_immediately():
    """직접(uses_agency=False) 선택 시 선정할 대상이 없으니 바로 통과한다."""
    state = _base_state()
    state["case_facts"] = {"_classified": True, "act_type": "일반수선"}
    state["disclosed_stage"] = {_STAGE_DOMAIN: 1}
    state["task_progress"] = {"uses_agency": False}
    directive = permit_phase_directive(state)
    assert "[Step 1-2]까지만" in directive


def test_stage1_architect_gate_optional_agency_needs_selection():
    """대행(uses_agency=True) 선택 시 architect_selected까지 확인돼야 한다."""
    state = _base_state()
    state["case_facts"] = {"_classified": True, "act_type": "일반수선"}
    state["disclosed_stage"] = {_STAGE_DOMAIN: 1}
    state["task_progress"] = {"uses_agency": True}
    directive = permit_phase_directive(state)
    assert "그 업체는 선정하셨어요" in directive

    state["task_progress"] = {"uses_agency": True, "architect_selected": True}
    directive = permit_phase_directive(state)
    assert "[Step 1-2]까지만" in directive


def test_max_allowed_stage_blocked_at_1_without_architect_decision():
    state = {
        "case_facts": {"_classified": True, "act_type": "신축"},
        "disclosed_stage": {_STAGE_DOMAIN: 1},
        "task_progress": {},
    }
    assert _max_allowed_stage(state) == 1


def test_construction_phase_no_longer_mentions_architect():
    """건축사 선정 안내는 이제 Step1-1 소관 - Step2 안내 문구엔 등장하면 안 된다."""
    state = _base_state()
    state["case_facts"] = {"_classified": True, "act_type": "신축"}
    state["disclosed_stage"] = {_STAGE_DOMAIN: 4, "construction_guide_shown": True, "construction_stage": 0}
    state["task_progress"] = {"application_submitted": True}
    directive = permit_phase_directive(state)
    assert "건축사" not in directive


def test_pending_stage2_confirmation_bypassed_when_interior_only():
    """구조 공사 없이 인테리어만 하면 착공신고ᆞ시공 게이트를 건너뛴다."""
    assert _pending_stage2_confirmation(1, {"interior_only": True}) is None
    assert _pending_stage2_confirmation(2, {"interior_only": True}) is None
    assert _pending_stage2_confirmation(1, {"interior_only": False}) == 1
```

파일 상단 import에 `_pending_stage2_confirmation` 사용을 위해 이미 있는 함수라 추가 import 불필요(같은 파일 내 정의).

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_agent_guard.py -k "stage1_architect or construction_phase_no_longer or pending_stage2_confirmation_bypassed" -v`
Expected: 여러 건 FAIL (`TypeError: _pending_stage1_confirmation() missing 1 required positional argument: 'case_facts'` 등)

- [ ] **Step 3: `src/roadmap.py` 수정**

`_pending_stage1_confirmation` 전체를 다음으로 교체:

```python
def _pending_stage1_confirmation(disclosed: int, task_progress: dict, case_facts: dict) -> int | None:
    """[Step 1-1]~[Step 1-3] 완료 확인이 아직 안 됐으면 그 단계 번호를 반환
    (이 이상 못 감), 다 확인됐으면 None.

    2026-08-04: Step0/Step1 재구성으로 [Step 1-1]의 의미가 "대상 판정"에서
    "건축사사무소 선정"으로 바뀌었다(판정 자체는 즉시 확정되는 사실이라 Step0으로
    옮기고 페이싱 대상에서 뺐다 - Step1-3이 설계도서를 만들려면 그 전에 건축사가
    정해져 있어야 하는데, 예전엔 이 선정이 Step2-1(착공 직전)에 있어 너무
    늦었다). 건축사 선정 게이트는 requires_licensed_architect 판정에 따라 갈린다:
      - True(의무): architect_selected 확인돼야 통과.
      - False(선택): uses_agency(직접/대행) 결정부터 필요 - 대행이면
        architect_selected까지, 직접이면 선정할 대상이 없어 바로 통과.
      - None(정보 부족): 게이트 없음(막지 않음).

    _max_allowed_stage(하드 캡)ᆞpermit_phase_directive(안내 문구)가 반드시 같은
    소스를 봐야 한다 - 따로 두면 한쪽만 고치고 잊어버리는 드리프트가 실제로
    있었다(2026-08-04, 세션 3d2f20a1에서 재현: 1-2 확인 전에 1-3 내용이 새는
    버그로 확인됨)."""
    if disclosed == 1:
        verdict = requires_licensed_architect(case_facts)
        if verdict is True:
            if not task_progress.get("architect_selected"):
                return 1
        elif verdict is False:
            uses_agency = task_progress.get("uses_agency")
            if uses_agency is None:
                return 1
            if uses_agency and not task_progress.get("architect_selected"):
                return 1
        # verdict is None -> 정보 부족, 막지 않음
    if disclosed == 2 and not task_progress.get("pre_diagnosis_checked"):
        return 2
    if disclosed == 3 and not task_progress.get("documents_prepared"):
        return 3
    return None
```

`_max_allowed_stage` 전체를 다음으로 교체:

```python
def _max_allowed_stage(state: "AgentState") -> int:
    """이번 턴 답변에 등장해도 되는 최대 [Step 1-N] 번호. 이 구조 자체가
    _STAGE_DOMAIN(permit) 전용이라, permit 판정 전에는 0(전부 금지) -
    signage/food/fire 얘기만 하는 대화는 애초에 permit 판정이 안 났으니
    항상 0으로 막혀서, 그 도메인들의 답변엔 [Step 1-N] 마커가 아예 등장하면
    안 된다(SYSTEM_PROMPT도 이렇게 지시). 판정 후에는 지금까지 공개된
    단계(disclosed_stage[_STAGE_DOMAIN]) + 1 - 한 턴에 최대 한 단계만 새로
    공개하도록 강제한다. 단, 앞 단계 완료 확인이 아직 안 됐으면
    (_pending_stage1_confirmation) 그 단계에서 더 못 올라간다."""
    if not state.get("case_facts", {}).get("_classified"):
        return 0
    disclosed = state.get("disclosed_stage", {}).get(_STAGE_DOMAIN, 0)
    case_facts = state.get("case_facts") or {}
    pending = _pending_stage1_confirmation(disclosed, state.get("task_progress") or {}, case_facts)
    if pending is not None:
        return pending
    return disclosed + 1
```

`_pending_stage2_confirmation` 전체를 다음으로 교체:

```python
def _pending_stage2_confirmation(construction_disclosed: int, task_progress: dict) -> int | None:
    """[Step 2-1]/[Step 2-2] 완료 확인이 아직 안 됐으면 그 단계 번호를 반환,
    다 확인됐으면 None. _pending_stage1_confirmation과 동일한 이유로 하드
    캡ᆞ안내 문구가 공유하는 단일 소스.

    2026-08-04: interior_only(구조 공사 없이 인테리어만 진행)면 착공신고ᆞ
    시공 게이트 자체를 건너뛴다 - 해당 없는 확인을 강제로 물어보게 되는
    문제를 막는다."""
    if task_progress.get("interior_only"):
        return None
    if construction_disclosed == 1 and not task_progress.get("construction_notice"):
        return 1
    if construction_disclosed == 2 and not task_progress.get("construction"):
        return 2
    return None
```

`permit_phase_directive` 함수 전체(`if not state.get("case_facts", {}).get("_classified"):`부터 함수 끝 `return "[진행 상태] Step 1~2(건축 인허가ᆞ공사) 안내가 모두 끝났습니다."`까지)를 다음으로 교체:

```python
    if not state.get("case_facts", {}).get("_classified"):
        return None

    disclosed_stage = state.get("disclosed_stage", {})
    disclosed = disclosed_stage.get(_STAGE_DOMAIN, 0)
    task_progress = state.get("task_progress") or {}
    case_facts = state.get("case_facts") or {}

    # 1-1(건축사 선정)ᆞ1-2(사전검토)ᆞ1-3(서류준비) 완료 확인 게이트 - 셋 다
    # _pending_stage1_confirmation 하나로 판단해서 하드 캡과 드리프트가 안 나게 한다.
    pending1 = _pending_stage1_confirmation(disclosed, task_progress, case_facts)
    if pending1 == 1:
        verdict = requires_licensed_architect(case_facts)
        if verdict is True:
            return (
                "[진행 상태] [Step 1-1: 건축사사무소 선정] 안내까지 끝났습니다. 이 사례는 "
                "건축법상 건축사사무소 설계가 필요한 대상(법적 의무)입니다 - 이 판정을 그대로 "
                "전달하고 재판단하거나 흐리게 말하지 마세요. 이번 턴에는 [Step 1-2]로 넘어가지 "
                "말고 \"건축사사무소는 선정하셨어요?\"처럼 완료 여부부터 확인하세요. 완료했다고 "
                "답하면 record_task_progress(architect_selected=True)로 기록한 뒤 다음 턴부터 "
                "[Step 1-2]를 안내하세요."
            )
        if verdict is False:
            uses_agency = task_progress.get("uses_agency")
            if uses_agency is None:
                return (
                    "[진행 상태] [Step 1-1: 건축사사무소 선정] 안내까지 끝났습니다. 이 사례는 "
                    "건축사사무소 설계가 법적 의무는 아닙니다(개인이 직접 진행 가능) - \"법적 "
                    "의무는 아닙니다\"라고 명확히 밝히세요. 이번 턴에는 [Step 1-2]로 넘어가지 "
                    "말고 \"직접 진행하실 건가요, 건축사사무소ᆞ행정사 등 대행업체 도움을 받으실 "
                    "건가요?\"처럼 물어보세요. 답변에 따라 record_task_progress(uses_agency=True "
                    "또는 False)로 기록한 뒤 다음 턴부터 이어가세요."
                )
            return (
                "[진행 상태] 대행업체 도움을 받기로 하셨습니다. 이번 턴에는 [Step 1-2]로 "
                "넘어가지 말고 \"그 업체는 선정하셨어요?\"처럼 완료 여부부터 확인하세요. "
                "완료했다고 답하면 record_task_progress(architect_selected=True)로 기록한 "
                "뒤 다음 턴부터 [Step 1-2]를 안내하세요."
            )
    if pending1 == 2:
        return (
            "[진행 상태] [Step 1-2: 사전 검토] 안내까지 끝났습니다. 이번 턴에는 [Step 1-3]"
            "(설계ᆞ서류 준비)로 넘어가지 말고 \"사전 검토 항목은 확인해 보셨어요?\"처럼 완료 "
            "여부부터 확인하세요. 완료했다고 답하면 "
            "record_task_progress(pre_diagnosis_checked=True)로 기록한 뒤 다음 턴부터 "
            "[Step 1-3]을 안내하세요."
        )
    if pending1 == 3:
        return (
            "[진행 상태] [Step 1-3: 설계ᆞ서류 준비] 안내까지 끝났습니다. 이번 턴에는 [Step 1-4]"
            "(신청ᆞ접수)로 넘어가지 말고 \"설계도서ᆞ서류 준비는 다 되셨어요?\"처럼 완료 여부부터 "
            "확인하세요. 완료했다고 답하면 record_task_progress(documents_prepared=True)로 "
            "기록한 뒤 다음 턴부터 [Step 1-4]를 안내하세요."
        )
    if disclosed < 4:
        return (
            f"[진행 상태] Step 1(건축 인허가)의 하위 단계 중 지금까지 [Step 1-{disclosed}]"
            f"까지 공개했습니다. 이번 턴에는 [Step 1-{disclosed + 1}]까지만 안내하고, 그 "
            f"이상은 절대 먼저 꺼내지 마세요 - 사용자가 이어서 요청하면 다음 턴에 공개하세요."
        )

    if not disclosed_stage.get("construction_guide_shown"):
        if not task_progress.get("application_submitted"):
            return (
                "[진행 상태] Step 1의 하위 단계 [Step 1-1]~[Step 1-4] 안내는 모두 끝났지만, "
                "신청ᆞ접수(Step 1-4에서 안내한 접수 절차)가 실제로 완료됐는지는 아직 확인되지 "
                "않았습니다. Step 2(공사)는 허가/신고 수리가 끝나야 실제로 진행할 수 있는 "
                "단계이니, 이번 턴에는 Step 2를 먼저 안내하지 말고 "
                "\"신청서 접수는 다 하셨어요?\"처럼 완료 여부부터 확인하세요. 완료했다고 "
                "답하면 record_task_progress(application_submitted=True)로 기록한 뒤 다음 "
                "턴부터 Step 2를 안내하세요."
            )
        return (
            "[진행 상태] Step 1의 하위 단계 [Step 1-1]~[Step 1-4] 안내가 모두 끝났고 신청ᆞ접수도 "
            "확인됐습니다. 이번 턴에는 사용자가 안 물어봐도 "
            "\"다음은 Step 2(공사) 단계입니다\"처럼 존재를 먼저 짚어주고 get_construction_guide를 "
            "호출해 안내하세요 - 식품위생ᆞ사업자등록 등 창업 준비 트랙으로 곧장 건너뛰지 마세요."
        )

    # Step 2 - 건축사 선정은 이제 Step1-1 소관이라 여기선 construction_notice/
    # construction/use_approval 게이트만 다룬다(2026-08-04).
    construction_disclosed = disclosed_stage.get("construction_stage", 0)
    pending2 = _pending_stage2_confirmation(construction_disclosed, task_progress)
    if pending2 == 1:
        return (
            "[진행 상태] [Step 2-1: 착공신고] 안내까지 끝났습니다. 이번 턴에는 [Step 2-2]로 "
            "넘어가지 말고 \"착공신고는 하셨어요?\"처럼 완료 여부부터 확인하세요. 완료했다고 "
            "답하면 record_task_progress(construction_notice=True)로 기록한 뒤 다음 턴부터 "
            "[Step 2-2]를 안내하세요."
        )
    if pending2 == 2:
        return (
            "[진행 상태] [Step 2-2: 시공ᆞ공사감리ᆞ소방시설ᆞ인테리어ᆞ장비 설치] 안내까지 "
            "끝났습니다. 이번 턴에는 [Step 2-3]으로 넘어가지 말고 \"시공은 끝나셨어요?\"처럼 "
            "완료 여부부터 확인하세요. 완료했다고 답하면 "
            "record_task_progress(construction=True)로 기록한 뒤 다음 턴부터 [Step 2-3]을 "
            "안내하세요."
        )
    if construction_disclosed < 3:
        return (
            f"[진행 상태] Step 2(공사)의 하위 단계 중 지금까지 "
            f"[Step 2-{construction_disclosed}]까지 공개했습니다. 이번 턴에는 "
            f"[Step 2-{construction_disclosed + 1}]까지만 안내하고, 그 이상은 절대 먼저 꺼내지 "
            "마세요 - 사용자가 이어서 요청하면 다음 턴에 공개하세요."
        )
    if not task_progress.get("use_approval"):
        return (
            "[진행 상태] Step 2(공사)의 하위 단계 [Step 2-1]~[Step 2-3] 안내가 모두 끝났습니다. "
            "이번 턴에는 \"사용승인은 받으셨어요?\"처럼 완료 여부부터 확인하세요. 완료했다고 "
            "답하면 record_task_progress(use_approval=True)로 기록하세요(구조 공사를 하는 "
            "케이스라면 착공신고ᆞ시공도 자동으로 완료 처리됩니다)."
        )

    return "[진행 상태] Step 1~2(건축 인허가ᆞ공사) 안내가 모두 끝났습니다."
```

`_roadmap_status_summary`의 `step2_done` 계산을 다음으로 교체(기존 `step2_fields = ...`~`step2_done = ...` 3줄):

```python
    step2_fields = ("construction_notice", "construction")
    step2_started = any(task_progress.get(f) for f in step2_fields) or bool(task_progress.get("use_approval"))
    # interior_only(구조 공사 없이 인테리어만)면 착공신고ᆞ시공은 해당 없음 -
    # 사용승인만으로 Step2 완료를 판단한다(2026-08-04).
    if task_progress.get("interior_only"):
        step2_done = bool(task_progress.get("use_approval"))
    else:
        step2_done = all(task_progress.get(f) for f in step2_fields) and bool(task_progress.get("use_approval"))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_agent_guard.py -v`
Expected: 이 Task에서 추가한 테스트는 PASS. `test_sequential_walkthrough_then_construction_transition`ᆞ`test_permit_phase_directive_states_architect_requirement_explicitly`ᆞ`test_no_force_when_construction_guide_already_shown` 등 기존 테스트는 새 1-1 의미와 충돌해서 FAIL할 수 있음 - **이 Task에서는 그대로 두고 Task 6에서 일괄 재작성**(지금 고치면 두 번 손대게 됨).

- [ ] **Step 5: Commit**

```bash
git add src/roadmap.py tests/test_agent_guard.py
git commit -m "feat: Step1-1을 건축사사무소 선정으로 재배치ᆞ직접/대행 게이트ᆞ인테리어전용 바이패스 추가"
```

---

## Task 3: `guard.py` - 툴콜 기반 게이트 위반 검증 추가

**Files:**
- Modify: `src/guard.py`
- Test: `tests/test_agent_guard.py`

**Interfaces:**
- Consumes: `src.roadmap._pending_stage1_confirmation(disclosed, task_progress, case_facts) -> int | None`(Task 2에서 시그니처 변경됨).
- Produces: `_tool_call_args_this_turn(messages, tool_name: str) -> list[dict]`, `_premature_stage_content_violations(state, messages) -> list[str]`.

**배경**: 마커([Step 1-N]) 기반 탐지는 LLM이 대괄호 헤더 없이 서술형으로만 다음 단계 내용을 전달하면 아예 못 잡는다(2026-08-04 스모크 테스트로 확인). `record_permit_synthesis`의 실제 호출 인자(`pre_diagnosis_items`/`required_documents`)는 텍스트 서식과 무관하게 항상 구조적으로 검사 가능하므로, 이걸 "그 단계 내용이 실제로 전달됐다"는 신호로 추가한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_agent_guard.py`에서 `test_synthesis_gap_still_triggers_retry` 함수 바로 다음에 추가:

```python
def test_premature_stage3_content_via_tool_call_without_marker():
    """회귀 테스트(세션 3d2f20a1, 2026-08-04) - [Step 1-2] 확인 전에
    record_permit_synthesis(required_documents=[...])가 호출되면, 텍스트에
    [Step 1-3] 마커가 전혀 없어도(서술형으로만 전달) 위반으로 잡아야 한다."""
    state = _base_state()
    state["disclosed_stage"] = {_STAGE_DOMAIN: 2}
    state["task_progress"] = {}
    call_id = "c1"
    state["messages"] = [
        HumanMessage(content="다음 단계 알려줘"),
        AIMessage(
            content="설계도서 작성은 몇 주 정도 걸립니다. 필요 서류는 다음과 같습니다...",
            tool_calls=[{
                "name": "record_permit_synthesis",
                "args": {"required_documents": ["용도변경 허가 신청서"]},
                "id": call_id,
            }],
        ),
        ToolMessage(content="기록됨.", tool_call_id=call_id, name="record_permit_synthesis"),
    ]
    update = guard_node(state)
    guard_msgs = [m for m in update.get("messages", []) if isinstance(m, ToolMessage) and m.name == "_answer_guard"]
    assert guard_msgs, "마커 없이 required_documents가 기록됐는데도 위반이 안 잡힘"
    assert "사전 검토" in guard_msgs[0].content


def test_premature_stage2_content_via_tool_call_without_marker():
    """건축사 선정(1-1) 확인 전에 pre_diagnosis_items가 기록되면 마커 유무와
    무관하게 위반으로 잡아야 한다."""
    state = _base_state()
    state["case_facts"] = {"_classified": True, "act_type": "신축"}
    state["disclosed_stage"] = {_STAGE_DOMAIN: 1}
    state["task_progress"] = {}
    call_id = "c1"
    state["messages"] = [
        HumanMessage(content="다음 단계 알려줘"),
        AIMessage(
            content="사전 검토 항목은 다음과 같습니다...",
            tool_calls=[{
                "name": "record_permit_synthesis",
                "args": {"pre_diagnosis_items": ["주차대수 확인"]},
                "id": call_id,
            }],
        ),
        ToolMessage(content="기록됨.", tool_call_id=call_id, name="record_permit_synthesis"),
    ]
    update = guard_node(state)
    guard_msgs = [m for m in update.get("messages", []) if isinstance(m, ToolMessage) and m.name == "_answer_guard"]
    assert guard_msgs, "건축사 선정 확인 전에 pre_diagnosis_items가 기록됐는데도 위반이 안 잡힘"


def test_premature_stage_content_allowed_when_gate_already_passed():
    """게이트가 이미 풀린 상태(pre_diagnosis_checked=True)라면 required_documents
    기록은 정상 경로이므로 위반이 아니다."""
    state = _base_state()
    state["disclosed_stage"] = {_STAGE_DOMAIN: 2}
    state["task_progress"] = {"pre_diagnosis_checked": True}
    call_id = "c1"
    state["messages"] = [
        HumanMessage(content="다음 단계 알려줘"),
        AIMessage(
            content="[Step 1-3: 설계ᆞ서류 준비] 필요 서류는 다음과 같습니다...",
            tool_calls=[{
                "name": "record_permit_synthesis",
                "args": {"required_documents": ["용도변경 허가 신청서"]},
                "id": call_id,
            }],
        ),
        ToolMessage(content="기록됨.", tool_call_id=call_id, name="record_permit_synthesis"),
    ]
    update = guard_node(state)
    assert not any(
        isinstance(m, ToolMessage) and m.name == "_answer_guard" for m in update.get("messages", [])
    )
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/test_agent_guard.py -k premature_stage -v`
Expected: FAIL(신규 위반 체크가 아직 없어 guard_msgs가 비어 있음)

- [ ] **Step 3: `src/guard.py` 수정**

상단 import를 다음으로 교체:

```python
from src.roadmap import (
    _STAGE_DOMAIN,
    _STEP2_KEYWORDS,
    _max_allowed_construction_stage,
    _max_allowed_stage,
    _mentioned_construction_stages,
    _mentioned_stages,
    _pending_stage1_confirmation,
)
```

`_synthesis_gap_violations` 함수 바로 다음에 새 함수 추가:

```python
def _tool_call_args_this_turn(messages, tool_name: str) -> list[dict]:
    """이번 사용자 턴(마지막 HumanMessage 이후) 안에서 tool_name으로 호출된
    tool_calls의 args 목록. _guard_retry_count와 같은 방식으로 역순 스캔한다."""
    args_list = []
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                if tc["name"] == tool_name:
                    args_list.append(tc["args"])
    return args_list


def _premature_stage_content_violations(state: "AgentState", messages) -> list[str]:
    """record_permit_synthesis가 이번 턴에 다음 단계 페이로드
    (pre_diagnosis_items/required_documents)를 채우려는데, 그 앞 단계 완료
    확인 게이트(_pending_stage1_confirmation)가 아직 안 풀려 있으면 위반.

    [Step 1-N] 마커 유무와 무관하게 툴 호출 자체로 판별한다 - 마커 기반 탐지는
    LLM이 대괄호 헤더 없이 서술형으로만 다음 단계 내용을 전달하면 아예 못
    잡는 사각지대가 있었다(2026-08-04, 세션 3d2f20a1에서 재현)."""
    case_facts = state.get("case_facts") or {}
    if not case_facts.get("_classified"):
        return []
    disclosed = state.get("disclosed_stage", {}).get(_STAGE_DOMAIN, 0)
    task_progress = state.get("task_progress") or {}
    pending = _pending_stage1_confirmation(disclosed, task_progress, case_facts)
    if pending is None:
        return []

    synth_calls = _tool_call_args_this_turn(messages, "record_permit_synthesis")
    violations = []
    if pending == 1 and any("pre_diagnosis_items" in a for a in synth_calls):
        violations.append(
            "건축사사무소 선정 확인 전에 [Step 1-2: 사전 검토] 내용(pre_diagnosis_items)을 "
            "기록했습니다. 먼저 건축사 선정(또는 직접/대행 결정) 확인부터 받으세요."
        )
    if pending == 2 and any("required_documents" in a for a in synth_calls):
        violations.append(
            "사전 검토 확인 전에 [Step 1-3: 설계ᆞ서류 준비] 내용(required_documents)을 "
            "기록했습니다. 먼저 사전 검토 완료 확인부터 받으세요."
        )
    return violations
```

`guard_node` 안 `violations.extend(_construction_guide_gap_violations(...))` 다음 줄에 추가:

```python
    violations.extend(_premature_stage_content_violations(state, messages))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_agent_guard.py -k "premature_stage" -v`
Expected: PASS (3건)

Run: `uv run pytest tests/test_agent_guard.py -v 2>&1 | tail -40`
Expected: Task 2에서 예상했던 실패(순서 재배치 관련 기존 테스트)만 남고 그 외 전부 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/guard.py tests/test_agent_guard.py
git commit -m "feat: 마커 없이 새는 단계 콘텐츠를 툴콜 인자 기반으로 탐지"
```

---

## Task 4: `prompts.py` - Step0/1 안내 문구 재작성

**Files:**
- Modify: `src/prompts.py`

**Interfaces:**
- Consumes: 없음(정적 텍스트, 실행 시점 상태는 `roadmap.py`의 동적 SystemMessage가 별도로 주입).

- [ ] **Step 1: `_ANSWER_PERMIT_STAGES`의 [Step 1-1] 불릿을 다음으로 교체**

기존(272~276행 근처):

```
- [Step 1-1: 대상 판정] 판정 결과ᆞ관할기관(관할 시ᆞ군ᆞ구청)ᆞ접수 방법을
  한두 줄로. 접수처ᆞ전자문서(온라인) 접수 가능 여부는 자동 검색된 법령 근거에서 확인해
  쓰세요("허가권자에게 제출, 전자문서 포함"). **도구 응답(현재 절차 결과 ...)에 건축사
  설계 의무 관련 문장이 함께 왔으면 그 문장을 그대로 전달하세요** - "개인이 직접
  가능하나 실무에선 인테리어 업체ᆞ행정사 도움을 받기도 한다"는 뉘앙스까지 포함해서
  (당신이 새로 판단하지 말고 온 문장을 옮기기).
```

새 텍스트:

```
- [Step 1-1: 건축사사무소 선정] 판정 결과(허가/신고/기재변경 등)ᆞ관할기관은 이미
  Step0(행위 유형 확인) 시점에 판정 도구 응답으로 전달됐으니 여기서 다시 설명하지
  마세요 - 이 단계는 오직 "건축사사무소를 선정해야 하는지ᆞ어떻게 하는지"만 다룹니다.
  매 턴 시스템 지시로 오는 "[건축사 설계 의무 판정]"류 문구를 그대로 전달하세요(당신이
  규모ᆞ공사 범위를 보고 재판단하거나 "필요할 수도 있다"처럼 흐리게 말하지 마세요):
  - 의무 대상이면: 왜 의무인지(설계도서를 건축사사무소가 작성해야 함) 한 줄 짚고
    선정을 안내하세요.
  - 의무가 아니면: "법적 의무는 아닙니다"라고 명확히 밝힌 뒤, 시스템 지시에 따라
    직접/대행 여부를 물어보세요. 대행을 선택하면 그 업체 선정까지 안내하고, 직접을
    선택하면 이 단계는 그냗로 끝(선정할 대상이 없음).
  건축사사무소 선정 여부는 이후 [Step 1-2]/[Step 1-3] 안내 방식(누가 진행하는지)에
  계속 영향을 줍니다.
```

- [ ] **Step 2: Step0 판정 안내 문구를 별도 문단으로 추가**

`_ANSWER_PERMIT_STAGES` 문자열 안에서 `**(B) permit 판정 완료(도구 응답으로 옴) 또는 permit 관련 질문**(정의` 로 시작하는 문단 바로 앞에 새 문단 삽입:

```
**permit 판정이 막 확정된 시점(도구 응답으로 옴)**: 이건 Step0(행위 유형+대상 판정)
소관입니다 - `[Step 1-N]` 헤더 없이 판정 결과ᆞ관할기관(관할 시ᆞ군ᆞ구청)ᆞ접수 방법을
한두 줄로 즉시 전달하세요("허가권자에게 제출, 전자문서 포함" 등은 자동 검색된 법령
근거에서 확인). 이 안내가 끝나면 곧장 이어서 [Step 1-1: 건축사사무소 선정]으로
넘어가세요(같은 턴에 붙여도 되고, 사용자가 응답을 기다리면 다음 턴에).

```

- [ ] **Step 3: [Step 1-3] 서류 목록의 역할 표기 재정비**

기존(296~316행 근처, `- [Step 1-3: 설계ᆞ서류 준비]`로 시작하는 전체 불릿)을 다음으로 교체:

```
- [Step 1-3: 설계ᆞ서류 준비] "설계도서 작성은 건축사사무소와 상담해서 준비하는
  과정이라 보통 몇 주가 걸립니다"처럼 시간이 걸리는 단계라는 감각을 먼저
  준 뒤, **"어디에 무엇을 제출할 것인지 위한 서류"를 실제 작성 주체 중심으로**
  안내하세요. **법적 명의자는 항상 건축주(신청인)이지만, 실제로 누가 작성하는지는
  Step1-1에서 결정된 직접/대행 여부(uses_agency)에 따라 다릅니다** - "건축사사무소가
  대행해준다"는 법적 요건이 아니라 대행을 선택했을 때만 해당하는 실무 관행입니다
  (법조문은 "신청서를 제출하는 자"만 규정하고 작성 주체는 명시하지 않음 -
  당신이 임의로 "보통 대행해준다"고 단정하지 마세요).
  **필요 서류를 하위 항목으로** 나열하되, 단순 나열이 아니라 **각 서류 옆에 실제
  작성 주체를 함께 표시**하세요. 다른 단계와 같은 마크다운 서식(굵게 `**소제목**`,
  목록은 `-` 불릿)을 그대로 쓰세요.
  예시 구조(uses_agency=False, 직접 진행인 경우 - 문구는 상황에 맞게):
  **[본인이 직접 작성ᆞ준비]**
  - 신고ᆞ신청서(별지 서식)
  - 대지 소유ᆞ사용권원 서류(등기부등본ᆞ임대차계약서 등)
  - 평면도ᆞ배치도 등 설계도서(건축사 의무 대상이 아니므로 직접 작성 가능, 부담되면
    인테리어 업체ᆞ행정사에 맡기는 경우도 있음)
  예시 구조(uses_agency=True, 대행업체 이용인 경우):
  **[건축주가 준비]**
  - 대지 소유ᆞ사용권원 서류(등기부등본ᆞ임대차계약서 등, 대행업체가 대신 발급받을
    수 없는 본인 명의 서류)
  **[대행업체(건축사사무소ᆞ행정사)가 작성ᆞ대행]**
  - 신고ᆞ신청서(별지 서식) - 법적 명의자는 건축주지만 작성ᆞ제출은 대행업체가 진행,
    건축주는 서명만
  - 평면도ᆞ배치도ᆞ내화ᆞ방화ᆞ피난ᆞ설비 도서 등 설계도서
  **[관공서 자체 확인]**
  - 용도변경 시 변경 전 평면도는 관공서가 건축물대장으로 직접 확인하므로
    본인이 준비할 필요 없음
  record_permit_synthesis(required_documents=[...])
```

- [ ] **Step 4: 문법 확인(구문 오류 없는지)**

Run: `uv run python -c "from src import prompts; print(len(prompts._ANSWER_PERMIT_STAGES))"`
Expected: 에러 없이 정수 출력(문자열 리터럴이 깨지지 않았는지 확인).

Run: `uv run pytest tests/test_system_prompt.py -v`
Expected: PASS(이 파일이 `_ANSWER_PERMIT_STAGES`가 `build_system_prompt` 결과에 포함되는지 등을 검증한다면 여전히 통과해야 함 - 실패 시 파일 열어서 어떤 문자열 매칭이 깨졌는지 확인 후 수정).

- [ ] **Step 5: Commit**

```bash
git add src/prompts.py
git commit -m "docs(prompt): Step1-1을 건축사사무소 선정으로ᆞ서류 역할표기를 직접/대행 기준으로 재작성"
```

---

## Task 5: `construction.py` - Step2 안내에서 건축사 문구 제거 + 구조공사 질문 + 소방시설 편입

**Files:**
- Modify: `src/agents/construction.py`

**Interfaces:**
- Consumes: `src.rag.regulation.search_regulations`(기존).
- Produces: `get_construction_guide() -> str`(쿼리 1개 추가로 반환 내용만 변경, 시그니처 동일).

- [ ] **Step 1: `_QUERIES`에 소방시설 관련 쿼리 추가**

```python
_QUERIES = [
    "착공신고는 어떻게 하나요?",
    "소방시설공사업자 완공검사증명서 발급 절차",
    "공사감리자 지정 기준",
    "사용승인을 받으려면 뭐가 필요한가요?",
]
```

(기존 "건축사 설계 의무와 예외" 쿼리는 제거 - 그 판정은 이제 Step1-1 소관이라 여기서 다시 검색할 필요 없음)

- [ ] **Step 2: docstring 전체를 다음으로 교체**

```python
@tool
def get_construction_guide() -> str:
    """착공신고ᆞ공사감리ᆞ소방시설ᆞ사용승인ᆞ인테리어ᆞ장비설치(Step 2: 공사) 절차를 반환합니다.

    - 착공신고를 어떻게ᆞ언제 하는지, 공사감리가 왜/언제 필요한지, 소방시설
      설치는 누가 하는지, 사용승인을 어떻게 받는지, 인테리어ᆞ장비 설치는
      어떻게 하는지 물으면 이 도구를 호출하세요(내부적으로 건축법ᆞ시행령ᆞ
      시행규칙을 여러 관점에서 검색 + 실무 체크리스트까지 합쳐서 돌려줍니다 -
      직접 다시 search_regulations를 호출할 필요 없습니다). 건축 인허가 판정
      ([Step 1-N])이 끝난 뒤에는 사용자가 따로 안 물어봐도 "다음은 공사
      단계"라는 걸 자연스럽게 짚어주면서 이 도구를 호출하세요 - Step 2를
      건너뛰고 바로 식품위생ᆞ사업자등록 등으로 넘어가면 안 됩니다.
    - **건축사사무소 선정 안내는 이제 Step1-1 소관입니다 - 여기서 다시
      꺼내지 마세요.** [Step 2-1]에서는 착공신고만 다룹니다.
    - **[Step 2-1: 착공신고] 안내 전에, 구조 공사(신축ᆞ증축ᆞ대수선 등)도
      하는지 인테리어만 진행하는지 먼저 물어보세요** - "구조 공사(뼈대ᆞ벽체
      변경 등)도 하시나요, 인테리어만 진행하시나요?"처럼. 인테리어만이라고
      답하면 record_task_progress(interior_only=True)로 기록하고, 착공신고ᆞ
      시공 항목은 해당 없음(N/A)으로 안내한 뒤 곧장 인테리어ᆞ장비 설치로
      넘어가세요. 구조 공사도 한다면 평소대로 착공신고부터 안내하세요.
    - **[Step 2-2] 소방시설 안내 시 담당 주체를 명확히 하세요**: 실제 설치는
      "소방시설공사업자"(일반 시공사와 다른 전문 면허업체 - 시공사ᆞ인테리어
      업체가 하청 주거나 별도 섭외)가 합니다. 건축허가ᆞ사용승인 절차상 소방서
      동의ᆞ확인은 관공서 간에 처리돼 사용자가 직접 챙길 일이 없지만,
      **완공검사증명서 발급은 챙겨야 하는 서류**입니다 - 대행업체(시공사ᆞ
      인테리어업체)를 쓰면 보통 그쪽이 챙기지만, 직접 진행한다면 사용자가
      시공 완료 후 소방서에 직접 신청해서 받아야 사용승인 신청(Step2-3)에
      첨부할 수 있다고 안내하세요.
    - 인테리어ᆞ장비 설치 항목(체크리스트 마지막 부분)은 법령 근거가 없는
      순수 실무 안내입니다 - [출처] 없이 안내된 그대로 전달하고, 없는
      법적 근거를 지어내 붙이지 마세요.
    - "이 사례에 감리가 꼭 필요한지"처럼 세부 요건이 여러 겹으로 갈리는
      판단은 검색 결과만으로 단정하지 말고, 의무처럼 단정하지 말고 원칙+예외를
      전달한 뒤 관할 구청 확인을 권하세요.
    - **각 단계(착공신고ᆞ시공ᆞ감리ᆞ소방시설ᆞ사용승인ᆞ인테리어) 안내 시 이 일을
      누가 하는지(소방시설공사업자ᆞ시공사ᆞ인테리어업체 / 건축주 본인) 구분해서
      안내하세요** - Step 1-3의 역할 표기와 같은 원칙.
    """
    logger.info("[도구] get_construction_guide()")
    legal_parts = "\n\n".join(search_regulations.func(q) for q in _QUERIES)
    return legal_parts + "\n\n" + _INTERIOR_NOTES
```

- [ ] **Step 3: 실제 검색 결과 확인**

Run: `uv run python -c "
from src.agents.construction import get_construction_guide
print(get_construction_guide.func())
" 2>&1 | grep -v "Warning\|Loading"`
Expected: 4개 검색 결과(착공신고ᆞ완공검사증명서ᆞ감리ᆞ사용승인)가 이어붙어 출력됨, 에러 없음. "완공검사증명서" 검색 결과가 실제로 관련 법조문을 반환하는지 눈으로 확인(빈 결과면 쿼리 문구 조정).

- [ ] **Step 4: Commit**

```bash
git add src/agents/construction.py
git commit -m "refactor: Step2 안내에서 건축사 문구 제거ᆞ구조공사 여부 질문ᆞ소방시설 역할 안내 추가"
```

---

## Task 6: `project_status.py` / `index.html` - Step0ᆞStep2ᆞStep3 UI 재배치

**Files:**
- Modify: `src/project_status.py`
- Modify: `src/static/index.html`
- Test: `tests/fixtures/roadmap_locks_check.js`, `tests/test_roadmap_locks.py`

**Interfaces:**
- Consumes: `taskProgress.interior_only`, `taskProgress.uses_agency`, `taskProgress.architect_selected`(Task 1에서 추가된 필드, JS에서는 그냥 객체 프로퍼티라 별도 선언 불필요).

- [ ] **Step 1: `src/project_status.py`의 `_step_substeps` 수정**

Step0 substep을 1개에서 2개로(행위 유형 + 판정), Step2 substep에 interior_only 반영. 함수 전체를 다음으로 교체:

```python
def _step_substeps(state: dict) -> list[list[tuple[str, bool]]]:
    """Step 0~4 각각의 [(라벨, 완료여부), ...]."""
    case_facts = state.get("case_facts") or {}
    permit_result = state.get("permit_result")
    food_result = state.get("food_result")
    fire_result = state.get("fire_result")
    signage_result = state.get("signage_result")
    tp = state.get("task_progress") or {}
    interior_only = bool(tp.get("interior_only"))

    return [
        [
            ("행위 유형 확인", bool(case_facts.get("act_type"))),
            ("허가ᆞ신고ᆞ기재변경 대상 판정", bool(permit_result)),
        ],
        [
            ("사전 검토", bool(tp.get("pre_diagnosis_checked"))),
            ("설계ᆞ서류 준비", bool(tp.get("documents_prepared"))),
            ("신청ᆞ접수", bool(tp.get("application_submitted"))),
        ],
        [
            ("착공신고", interior_only or bool(tp.get("construction_notice"))),
            ("공사(시공)", interior_only or bool(tp.get("construction"))),
            ("사용승인", bool(tp.get("use_approval"))),
        ],
        [
            ("식품위생 영업신고 확인", food_result is not None),
            ("간판ᆞ옥외광고물 확인", signage_result is not None),
            ("사업자등록", bool(tp.get("business_registration"))),
            ("위생교육 이수", bool(tp.get("hygiene_education"))),
        ],
        [
            ("인테리어ᆞ장비 설치", bool(tp.get("interior_equipment"))),
            ("직원 등록(4대보험 가입)", bool(tp.get("staff_registration")) or tp.get("hires_staff") is False),
            ("영업 시작", bool(tp.get("opened"))),
        ],
    ]
```

(주의: Step1의 substep 개수가 4→3으로 줄었다 - "허가ᆞ신고 대상 판정"이 Step0으로 옮겨갔으므로. Step3에서 "소방시설 확인"이 빠졌다 - Step2로 옮겨서 `fire_result`는 더 이상 이 함수에서 쓰지 않으니 상단의 `fire_result = state.get("fire_result")` 줄도 삭제한다. 위 코드에 이미 반영됨.)

- [ ] **Step 2: `_track_status`가 여전히 정상 동작하는지 확인**

`_track_status`ᆞ`compute_project_status`는 `steps` 리스트의 인덱스(`_CONSTRUCTION_TRACK_INDICES = (0, 1, 2)`, `_STARTUP_TRACK_INDICES = (3, 4)`)를 그대로 참조하므로 substep 개수가 바뀌어도 로직 변경 불필요 - 그대로 둔다.

- [ ] **Step 3: `src/static/index.html`의 Step0/Step2/Step3 배열 수정**

`steps` 배열 정의(941행 근처) 전체를 다음으로 교체(`step1Substeps`ᆞ`step3AllDone` 계산부는 아래 Step 4~5에서 별도로 손본 뒤 이 배열이 그 값을 참조):

```javascript
    const steps = [
      {
        title: "Step 0 · 건축 유형 + 대상 판정",
        substeps: [
          { text: actType || "행위 유형 파악", done: !!actType },
          {
            text: latestPermitResult ? `판정: ${latestPermitResult.permit_type}` : (caseTypeResult ? `판정: ${caseTypeResult}` : "허가ᆞ신고ᆞ기재변경 대상 판정"),
            done: classified,
          },
        ],
        architect: requiresArchitect,
      },
      {
        title: "Step 1 · 건축 인허가 실무",
        substeps: step1Substeps,
      },
      {
        title: "Step 2 · 공사",
        substeps: [
          {
            text: "착공신고", done: !!taskProgress.construction_notice, field: "construction_notice",
            notApplicable: !!taskProgress.interior_only,
            naReason: "구조 공사 없이 인테리어만 진행하는 경우라 착공신고 대상이 아닙니다",
          },
          {
            text: "시공", done: !!taskProgress.construction, field: "construction",
            notApplicable: !!taskProgress.interior_only,
            naReason: "구조 공사 없이 인테리어만 진행하는 경우라 해당 없습니다",
          },
          {
            text: latestFireResult && latestFireResult.length ? `소방시설: ${latestFireResult.join("ᆞ")}` : "소방시설",
            done: latestFireResult != null,
            notApplicable: Array.isArray(latestFireResult) && latestFireResult.length === 0,
            naReason: "연면적 기준상 설치 대상 소방시설이 없습니다",
            locked: !taskProgress.construction && !taskProgress.interior_only,
            lockReason: "공사(시공) 완료 후 설치ᆞ확인 가능",
          },
          { text: "인테리어ᆞ장비 설치", done: !!taskProgress.interior_equipment, field: "interior_equipment" },
          { text: "사용승인", done: !!taskProgress.use_approval, field: "use_approval" },
        ],
      },
      {
        title: "Step 3 · 창업 행정",
        parallel: true,
        substeps: [
          {
            text: latestFoodResult ? `식품위생: ${latestFoodResult}` : "식품위생 영업신고",
            done: latestFoodResult != null,
            notApplicable: latestFoodResult === "해당없음" || latestFoodResult === "신고대상제외",
            naReason: "식품위생법상 별도 영업신고 대상이 아닙니다",
            locked: !taskProgress.use_approval,
            lockReason: "사용승인 완료 후 신고 가능",
          },
          {
            text: latestSignageResult ? `간판: ${latestSignageResult}` : "간판ᆞ옥외광고물",
            done: latestSignageResult != null,
            notApplicable: latestSignageResult != null && latestSignageResult.includes("불필요"),
            naReason: "허가ᆞ신고가 필요 없는 간판입니다",
            locked: !taskProgress.construction && !taskProgress.interior_only,
            lockReason: "공사(시공) 완료 후 설치 가능",
          },
          {
            text: "사업자등록",
            done: !!taskProgress.business_registration,
            field: "business_registration",
            locked: !((taskProgress.documents_prepared || permitNotNeeded) && latestFoodResult != null),
            lockReason: "임대차계약서ᆞ영업신고 확인 후 진행 가능",
          },
          {
            text: "위생교육",
            done: !!taskProgress.hygiene_education,
            field: "hygiene_education",
            notApplicable: latestFoodResult === "해당없음" || latestFoodResult === "신고대상제외",
            naReason: "식품위생법상 영업자가 아니라 위생교육 대상이 아닙니다",
          },
        ],
      },
      {
        title: "Step 4 · 오픈 준비",
        substeps: [
          {
            text: "직원 등록(4대보험 가입)",
            done: !!taskProgress.staff_registration || taskProgress.hires_staff === false,
            notApplicable: taskProgress.hires_staff === false,
            naReason: "직원을 두지 않아 4대보험 가입 대상이 아닙니다",
            field: "staff_registration",
            locked: !taskProgress.business_registration,
            lockReason: "사업자등록 완료 후 진행 가능",
          },
          {
            text: "영업 시작",
            done: !!taskProgress.opened,
            field: "opened",
            locked: !step3AllDone,
            lockReason: "식품위생ᆞ간판ᆞ사업자등록ᆞ위생교육 완료 후 가능",
          },
        ],
      },
    ];
```

(소방시설 항목이 Step3에서 빠지고 Step2로 이동했다. Step2의 "소방시설" 잠금 조건에 `!taskProgress.interior_only`를 추가한 이유: 인테리어 전용 케이스는 `construction`이 계속 false로 남을 수 있어(구조 공사 자체가 없으니), 그 경우 소방시설 잠금이 영원히 안 풀리면 안 된다.)

- [ ] **Step 4: `step3AllDone` 재계산(소방시설 제외)**

기존(935~939행 근처) `const step3AllDone = ...` 전체를 다음으로 교체:

```javascript
    // Step 4 "영업 시작" 잠금 기준 - 소방시설은 Step2로 옮겨갔으므로
    // step3AllDone에서 제외(이중 게이트 방지, 2026-08-04). 나머지 4항목
    // (식품위생ᆞ간판ᆞ사업자등록ᆞ위생교육)은 그대로.
    const step3AllDone = latestFoodResult != null
      && latestSignageResult != null
      && !!taskProgress.business_registration
      && !!taskProgress.hygiene_education;
```

- [ ] **Step 5: Step1 substeps 정의부(step1Substeps 만드는 코드)에서 "허가ᆞ신고 대상 확인" 항목 제거**

`step1Substeps`를 만드는 코드(908행 근처, `const step1Substeps = [{ text: "허가ᆞ신고 대상 확인", done: permitStage >= 2, tags: factInfoLines }];`로 시작)에서 이 첫 줄을 제거하고 빈 배열로 시작하도록 변경:

```javascript
    const step1Substeps = [];
    if (factInfoLines.length) {
      // 판정 입력값(연면적ᆞ용도지역 등 참고 정보)은 Step0로 옮긴 판정 항목이
      // 아니라 별도 참고 줄로 유지 - 첫 substep으로 안 넣고 info 줄로만.
      step1Substeps.push({ text: factInfoLines.join(" · "), info: true });
    }
```

(이 변경으로 `permitStage`/`computeCurrentStage` 관련 로직이 "몇 번째부터 done인지" 세는 방식이라 첫 항목이 사라지면 인덱스가 밀린다 - `buildChecklists()`가 반환하는 `checklists` 배열 자체는 안 건드리므로 `computeCurrentStage`는 그대로 동작한다. 단, `factInfoLines`를 `info: true`로 넣으면 done/undone 판정에 안 끼므로 안전하다.)

- [ ] **Step 6: JS 하네스 테스트 갱신**

`tests/fixtures/roadmap_locks_check.js`에서 `buildSteps`의 `Function` 파라미터 목록과 호출부에 `interiorOnly`를 추가할 필요는 없다 - `taskProgress` 객체 안에 `interior_only` 키를 넣으면 그대로 전달되므로(이미 `taskProgress` 전체를 넘기는 구조). 파일 끝(`process.stdout.write(JSON.stringify(results));` 바로 앞)에 새 검증 블록 추가:

```javascript
// 인테리어 전용 케이스 - 착공신고ᆞ시공이 notApplicable로 닫히는지(2026-08-04)
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: { interior_only: true } });
  check("interior_only: 착공신고 해당없음", !!findSub(steps, "Step 2 · 공사", "착공신고").notApplicable);
  check("interior_only: 시공 해당없음", !!findSub(steps, "Step 2 · 공사", "시공").notApplicable);

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: ["소화기구"], latestSignageResult: null, taskProgress: { interior_only: true } });
  check("interior_only: 소방시설은 construction 없이도 잠금 해제", !isVisuallyLocked(findSub(steps, "Step 2 · 공사", "소방시설")));
}
```

(주의: `findSub(steps, "Step 2 · 공사", "소방시설")`처럼 이제 소방시설이 "Step 2 · 공사"에 있으므로 stepTitle 인자를 그렇게 바꿔서 찾는다. 기존 "fire: 공사 전엔 잠김"류 테스트도 stepTitle을 `"Step 3 · 창업 행정"`에서 `"Step 2 · 공사"`로 바꿔야 한다 - 아래 Step 7 참고.)

기존 소방시설 관련 블록(56행 근처 `// 소방시설ᆞ간판 - 공사(시공) 전/후`)을 다음으로 교체 - 소방시설과 간판을 분리(소방시설은 Step2, 간판은 Step3):

```javascript
// 소방시설(Step2) - 공사(시공) 전/후
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: ["소화기구"], latestSignageResult: "신고", taskProgress: { construction: false } });
  check("fire: 공사 전엔 잠김", isVisuallyLocked(findSub(steps, "Step 2 · 공사", "소방시설")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: ["소화기구"], latestSignageResult: "신고", taskProgress: { construction: true } });
  check("fire: 공사 후엔 해제", !isVisuallyLocked(findSub(steps, "Step 2 · 공사", "소방시설")));
}

// 간판(Step3) - 공사(시공) 전/후
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: "신고", taskProgress: { construction: false } });
  check("signage: 공사 전엔 잠김", isVisuallyLocked(findSub(steps, "Step 3 · 창업 행정", "간판")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: "신고", taskProgress: { construction: true } });
  check("signage: 공사 후엔 해제", !isVisuallyLocked(findSub(steps, "Step 3 · 창업 행정", "간판")));
}
```

"na: 소방시설 대상 없음(빈 배열)이면 해당없음"ᆞ"na: 소방시설 있으면 해당없음 아님" 블록(129~135행 근처)의 `findSub(steps, "Step 3 · 창업 행정", "소방시설")`도 전부 `"Step 2 · 공사"`로 교체.

- [ ] **Step 7: `tests/test_roadmap_locks.py` 갱신**

`test_fire_and_signage_locked_until_construction`을 두 개로 분리:

```python
def test_fire_locked_until_construction(roadmap_lock_results):
    assert roadmap_lock_results["fire: 공사 전엔 잠김"]
    assert roadmap_lock_results["fire: 공사 후엔 해제"]


def test_signage_locked_until_construction(roadmap_lock_results):
    assert roadmap_lock_results["signage: 공사 전엔 잠김"]
    assert roadmap_lock_results["signage: 공사 후엔 해제"]


def test_interior_only_closes_construction_substeps(roadmap_lock_results):
    assert roadmap_lock_results["interior_only: 착공신고 해당없음"]
    assert roadmap_lock_results["interior_only: 시공 해당없음"]
    assert roadmap_lock_results["interior_only: 소방시설은 construction 없이도 잠금 해제"]
```

`test_not_applicable_items_are_closed`에서 `roadmap_lock_results["na: 소방시설 대상 없음(빈 배열)이면 해당없음"]`ᆞ`roadmap_lock_results["na: 소방시설 있으면 해당없음 아님"]` 참조는 라벨 문자열이 안 바뀌었으므로 그대로 둔다(찾는 Step 타이틀만 fixture 쪽에서 바뀜, 테스트 파일은 라벨만 참조하므로 영향 없음).

- [ ] **Step 8: 테스트 통과 확인**

Run: `uv run pytest tests/test_roadmap_locks.py tests/test_markdown_render.py -v`
Expected: PASS 전체.

- [ ] **Step 9: Commit**

```bash
git add src/project_status.py src/static/index.html tests/fixtures/roadmap_locks_check.js tests/test_roadmap_locks.py
git commit -m "refactor: 로드맵 UI 재배치 - Step0에 판정 추가ᆞ소방시설 Step2 이동ᆞ인테리어전용 N/A"
```

---

## Task 7: `test_agent_guard.py` 전체 정합성 재작성

**Files:**
- Modify: `tests/test_agent_guard.py`

Task 2~3에서 이미 새 테스트를 추가했지만, **기존** 테스트 중 옛 1-1(대상판정) 의미ᆞ옛 `_pending_stage1_confirmation` 2-인자 시그니처를 가정하는 것들이 깨져 있다. 하나씩 정리한다.

- [ ] **Step 1: `test_sequential_walkthrough_then_construction_transition` 재작성**

전체를 다음으로 교체(신축ᆞ건축사 의무 대상 케이스로 1-1(건축사선정)→1-2(사전검토)→1-3(서류)→1-4(접수)→Step2 순서 검증):

```python
def test_sequential_walkthrough_then_construction_transition():
    """건축사 의무 대상(신축) 케이스로 1-1(건축사 선정)→1-2(사전검토)→1-3(서류
    준비)→1-4(신청접수) 순서로 진행하고, 다 끝나면 Step2(공사) 전환 넛지가
    나오고, get_construction_guide 호출 후엔 Step 2-1→2-2→2-3 순서로 같은
    완료-확인 게이트가 적용된다.

    2026-08-04: Step0/Step1 재구성으로 [Step 1-1]의 의미가 "대상 판정"에서
    "건축사사무소 선정"으로 바뀌었다 - 판정 자체(허가/신고/기재변경)는 Step0에서
    이미 끝난 것으로 간주하고 이 테스트는 Step1 내부(건축사 선정~신청접수)만
    검증한다."""
    state = _base_state()
    state["case_facts"] = {"_classified": True, "act_type": "신축"}

    directive = permit_phase_directive(state)
    assert "건축사사무소는 선정하셨어요" in directive, "신축은 건축사 의무 대상(True)"

    state["messages"].append(AIMessage(content="[Step 1-1: 건축사사무소 선정] 건축사 설계가 필요합니다."))
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 1
    directive = permit_phase_directive(state)
    assert "건축사사무소는 선정하셨어요" in directive, "선정 확인 전엔 1-2로 못 감"
    state["task_progress"] = {"architect_selected": True}
    assert "[Step 1-2]까지만" in permit_phase_directive(state)

    state["messages"].append(AIMessage(content="[Step 1-2: 사전 검토] 주차대수 확인 필요"))
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 2

    directive = permit_phase_directive(state)
    assert "사전 검토 항목은 확인해 보셨어요" in directive
    state["task_progress"] = {**state["task_progress"], "pre_diagnosis_checked": True}
    assert "[Step 1-3]까지만" in permit_phase_directive(state)

    state["messages"].append(AIMessage(content="[Step 1-3: 설계ᆞ서류 준비] 건축ᆞ대지 현황도"))
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 3

    directive = permit_phase_directive(state)
    assert "서류 준비는 다 되셨어요" in directive
    state["task_progress"] = {**state["task_progress"], "documents_prepared": True}
    assert "[Step 1-4]까지만" in permit_phase_directive(state)

    state["messages"].append(
        AIMessage(content="[Step 1-4: 신청ᆞ접수] 관할 구청에 접수합니다. 예상 소요 기간은 신고 기준 3~5일입니다.")
    )
    state = _apply(state, guard_node(state))
    assert _disclosed(state) == 4

    directive = permit_phase_directive(state)
    assert "신청서 접수는 다 하셨어요" in directive

    state["task_progress"] = {**state["task_progress"], "application_submitted": True}
    directive = permit_phase_directive(state)
    assert "신청ᆞ접수도 확인됐습니다" in directive

    call_id = "c1"
    state["messages"].append(
        AIMessage(content="", tool_calls=[{"name": "get_construction_guide", "args": {}, "id": call_id}])
    )
    state["messages"].append(
        ToolMessage(content="[착공ᆞ공사 단계 안내]...", tool_call_id=call_id, name="get_construction_guide")
    )
    state["messages"].append(AIMessage(content="다음은 Step 2(공사) 단계입니다. 착공신고부터 안내드릴게요."))
    state = _apply(state, guard_node(state))

    assert state["disclosed_stage"].get("construction_guide_shown") is True

    directive = permit_phase_directive(state)
    assert "[Step 2-1]까지만" in directive
    assert "건축사" not in directive, "건축사 선정 안내는 이제 Step1-1 소관"

    state["messages"].append(AIMessage(content="[Step 2-1: 착공신고] 착공신고 방법 안내드릴게요."))
    state = _apply(state, guard_node(state))
    assert state["disclosed_stage"].get("construction_stage") == 1

    directive = permit_phase_directive(state)
    assert "착공신고는 하셨어요" in directive
    state["task_progress"] = {**state["task_progress"], "construction_notice": True}
    assert "[Step 2-2]까지만" in permit_phase_directive(state)

    state["messages"].append(AIMessage(content="[Step 2-2: 시공ᆞ공사감리ᆞ소방시설ᆞ인테리어ᆞ장비 설치] 시공 안내드릴게요."))
    state = _apply(state, guard_node(state))
    assert state["disclosed_stage"].get("construction_stage") == 2

    directive = permit_phase_directive(state)
    assert "시공은 끝나셨어요" in directive
    state["task_progress"] = {**state["task_progress"], "construction": True}
    assert "[Step 2-3]까지만" in permit_phase_directive(state)

    state["messages"].append(AIMessage(content="[Step 2-3: 사용승인] 사용승인 절차 안내드릴게요."))
    state = _apply(state, guard_node(state))
    assert state["disclosed_stage"].get("construction_stage") == 3

    directive = permit_phase_directive(state)
    assert "사용승인은 받으셨어요" in directive
    state["task_progress"] = {**state["task_progress"], "use_approval": True}
    assert permit_phase_directive(state) == "[진행 상태] Step 1~2(건축 인허가ᆞ공사) 안내가 모두 끝났습니다."
```

- [ ] **Step 2: `_base_state`에서 `case_facts`에 `act_type` 없어도 되는 다른 테스트들 확인**

`_base_state()`가 `{"case_facts": {"_classified": True}, ...}`를 반환하는데(act_type 없음), 이 상태로 `permit_phase_directive`를 호출하면 `disclosed=0`일 때 `_pending_stage1_confirmation(0, ..., case_facts)`는 disclosed==1 분기에 안 걸리니 안전하다(0은 그대로 통과). `test_revisit_lowers_stage_instead_of_skipping`ᆞ`test_step1_5_marker_not_recognized`ᆞ`test_unrelated_answer_does_not_touch_disclosed_stage`는 `disclosed_stage`를 2~4로 직접 세팅하고 시작하므로 disclosed==1 분기를 안 거친다 - **수정 불필요**, 그대로 둔다.

`test_construction_guide_gap_triggers_retry`ᆞ`test_construction_guide_gap_ignored_before_application_submitted`ᆞ`test_construction_guide_called_suppresses_gap_violation`ᆞ`test_construction_guide_gap_ignores_unrelated_topics`ᆞ`test_force_construction_guide_when_ai_just_proposed_step2`ᆞ`test_no_force_before_application_submitted_even_if_ai_proposed_step2`ᆞ`test_no_force_when_construction_guide_already_shown`ᆞ`test_no_force_when_step1_not_finished`ᆞ`test_no_force_on_unrelated_topic_even_if_step1_done`ᆞ`test_no_force_when_last_message_is_not_human`는 전부 `disclosed_stage = {_STAGE_DOMAIN: 4}`(또는 1)로 시작해서 `_should_force_construction_guide`/`_construction_guide_gap_violations`만 검증하므로 **수정 불필요**.

- [ ] **Step 3: `test_permit_phase_directive_states_architect_requirement_explicitly` 삭제**

이 테스트는 "Step2 진입 시 건축사 의무 문구가 뜬다"를 검증했는데, 이제 그 문구는 Step1-1에서 뜬다(Task 2의 `test_stage1_architect_gate_mandatory_blocks_until_selected`ᆞ`test_stage1_architect_gate_optional_asks_direct_or_agency`가 이미 그 역할을 대체). 이 함수 전체를 삭제한다.

- [ ] **Step 4: 전체 실행**

Run: `uv run pytest tests/test_agent_guard.py -v 2>&1 | tail -60`
Expected: 전체 PASS. FAIL이 남으면 실패 메시지를 읽고 새 1-1(건축사선정) 의미와 충돌하는 지점을 위 패턴대로 수정.

- [ ] **Step 5: Commit**

```bash
git add tests/test_agent_guard.py
git commit -m "test: Step1 재배치에 맞춰 순차 진행ᆞ건축사 게이트 테스트 정합성 복구"
```

---

## Task 8: 전체 검증 + LLM 스모크 테스트

**Files:** 없음(검증 전용)

- [ ] **Step 1: 전체 유닛 테스트**

Run: `uv run pytest -q`
Expected: 전부 PASS. 실패하면 해당 파일로 돌아가 원인 수정(대부분 Task 2/6에서 다룬 마커ᆞ필드 이름 불일치일 것).

- [ ] **Step 2: LLM 스모크 테스트 스크립트 작성**

`/private/tmp/claude-501/.../scratchpad/smoke_role_redesign.py`(스크래치패드 경로, 세션마다 다르므로 실행 시점의 실제 스크래치패드 경로 사용):

```python
"""역할 기반 재설계 실측 스모크 테스트 (실제 LLM 호출) - 4개 시나리오."""
from __future__ import annotations
import uuid
from src.graph import graph
from src.pipeline import query
from src.roadmap import _STAGE_DOMAIN

def run_scenario(name, case_facts, permit_result):
    user_id = f"smoke-{uuid.uuid4().hex[:8]}"
    config_dict = {"configurable": {"thread_id": user_id}}
    graph.update_state(config_dict, {
        "case_facts": case_facts,
        "permit_result": permit_result,
        "disclosed_stage": {_STAGE_DOMAIN: 0},
        "task_progress": {},
    })
    print("=" * 60, f"\n[{name}] 턴1: 다음 안내해줘")
    r1 = query("다음 단계 안내해줘", user_id=user_id)
    print(r1["answer"])
    return user_id, config_dict

# 시나리오 1: 신축(건축사 의무 대상=True)
run_scenario("신축(의무)", {"_classified": True, "act_type": "신축"}, "건축허가")

# 시나리오 2: 일반수선(의무 아님=False) - 직접/대행 분기 확인
run_scenario("일반수선(선택)", {"_classified": True, "act_type": "일반수선"}, "인허가불필요")
```

- [ ] **Step 3: 서버 없이 스크립트 실행(그래프 직접 호출)**

Run: `uv run python <스크래치패드 경로>/smoke_role_redesign.py 2>&1 | grep -v "Warning\|Loading"`
Expected:
- 시나리오 1 답변에 "건축사사무소" 선정 관련 문구가 있고 "법적 의무" 또는 "필요한 대상"이 명시돼야 함.
- 시나리오 2 답변에 "직접 진행하실 건가요" 또는 "대행업체" 관련 질문이 있어야 함.
- 두 시나리오 모두 답변에 [Step 1-2]/[Step 1-3] 서류ᆞ사전검토 상세 내용이 **먼저** 나오면 안 됨(1-1 게이트가 안 풀렸으므로).

- [ ] **Step 4: 인테리어 전용 케이스 확인**

동일 스크립트에 시나리오 3 추가해서 실행:

```python
# 시나리오 3: 용도변경(구조공사 없이 인테리어만) - Step2 진입 후 확인
user_id, config_dict = run_scenario(
    "용도변경(인테리어전용)",
    {"_classified": True, "act_type": "용도변경", "current_facility_group": 8, "desired_facility_group": 7, "size_sqm": 80},
    "용도변경허가",
)
graph.update_state(config_dict, {
    "disclosed_stage": {_STAGE_DOMAIN: 4, "construction_guide_shown": True, "construction_stage": 0},
    "task_progress": {"application_submitted": True},
})
print("=" * 60, "\n[인테리어전용] 턴2: 공사 단계 알려줘")
r2 = query("구조 공사는 안 하고 인테리어만 할 거야", user_id=user_id)
print(r2["answer"])
print("-- task_progress:", r2.get("task_progress"))
```

Expected: `task_progress`에 `interior_only: True`가 반영되고, 착공신고ᆞ시공을 "해당 없음"으로 안내.

- [ ] **Step 5: 결과를 실제로 읽고 판단** (자동 assert 아님 - 사람이 답변 내용을 읽고 의도대로 나왔는지 확인)

CLAUDE.md §3 원칙: 유닛 테스트 통과만으로 완료 보고하지 않는다. 스모크 테스트 출력을 읽고 다음을 확인:
1. 판정 안내가 Step0에서 헤더 없이 즉시 나오는지(1-1 이전에).
2. 건축사 의무/선택 분기가 실제로 다르게 응답되는지.
3. 게이트 미확인 상태에서 다음 단계 상세 내용이 새지 않는지.

문제 발견 시 해당 파일(대부분 `roadmap.py` 게이트 로직 또는 `prompts.py` 문구)로 돌아가 수정 후 재실행.

- [ ] **Step 6: 스크래치패드 스크립트 정리**

Run: `rm <스크래치패드 경로>/smoke_role_redesign.py`

(커밋 없음 - 검증 전용 Task)

---

## Self-Review 체크리스트 (계획 작성자용, 참고)

- [x] Spec의 6개 결정 사항(Step0 확장ᆞ건축사 선정 이동ᆞ직접/대행 분기ᆞ서류 역할표기ᆞ소방시설 이동ᆞ시공 N/A ᆞ게이트 신뢰성) 모두 Task로 매핑됨(Task 2,4,5,6이 대부분 분담, Task 3이 게이트 신뢰성 전담).
- [x] 새 상태 필드 3개(uses_agency/architect_selected/interior_only) 전부 Task 1에서 정의되고 이후 Task에서 소비됨.
- [x] "TBD"/"나중에"류 표현 없음 - 모든 코드 블록 완성형.
- [x] 함수 시그니처 일관성: `_pending_stage1_confirmation(disclosed, task_progress, case_facts)` - Task 2에서 정의, Task 3(guard.py)ᆞTask 7(테스트)에서 동일 시그니처로 소비.
