"""로드맵 단계 추적 + 진행 안내 (agent.py에서 분리, Phase 4).

2026-08-05: `[Step 1-N]`/`[Step 2-N]` 텍스트 마커 + `disclosed_stage` 기반
페이싱ᆞguard.py의 마커 개수 강제 검증을 통째로 제거하고, `current_step_directive`
로 교체했다(사용자 요청 - "다음 단계만 계속 알려주지 말고, 현재 상태를 확인하고
체크리스트를 한 단계씩 진행하도록"). 새 방식은 매 턴
`roadmap_model.compute_roadmap_steps()`(Step1~9의 단일 계산 소스)가 계산한
"지금 여기" Step의 미완료ᆞ비잠금 substep 목록을 그대로 SystemMessage로
전달하고, 강제 장치 없이 프롬프트 지시로만 페이싱을 유도한다(브레인스토밍에서
사용자가 명시적으로 선택한 트레이드오프 - 강한 강제 대신 간결한 코드).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.roadmap_model import (
    business_registration_unlocked,
    compute_roadmap_steps,
    fire_signage_unlocked,
    food_report_unlocked,
    is_visually_locked,
)

if TYPE_CHECKING:
    from src.agent import AgentState

# 건축 트랙(순차 - Step1~7)과 창업 준비 트랙(병렬 가능 - Step8~9)의 경계.
# project_status.py의 5그룹 트랙 분리(_CONSTRUCTION_TRACK_INDICES/
# _STARTUP_TRACK_INDICES)와 같은 기준.
_CONSTRUCTION_TRACK_MAX = 7
_STARTUP_TRACK_MIN = 8

# Step 번호별 배경 설명 - 예전 [Step 1-N]/[Step 2-N] 안내문에 있던 실질적인
# 지침(왜 이 단계를 하는지ᆞ누가 하는지ᆞ소요 기간 등)을 마커ᆞ강제 없이
# 그대로 옮겨왔다. "현재 단계"로 계산된 Step에만 붙는다 - 예전엔 permit 블록이
# 켜지면 이 지침 전부가 상시 주입됐는데, 이제 활성 단계 하나(많아야 둘, 병렬
# 트랙 포함)만 붙어서 토큰도 준다.
STEP_GUIDANCE: dict[int, str] = {
    3: (
        "건축사사무소 설계 의무 여부는 [건축사 설계 의무 판정] 시스템 지시를 그대로 "
        "전달하세요(스스로 규모ᆞ공사 범위를 보고 재판단하거나 흐리게 말하지 마세요). "
        "의무 대상이면 왜 의무인지(설계도서를 건축사사무소가 작성해야 함) 짚고 선정을 "
        "안내하고, 의무가 아니면 \"법적 의무는 아닙니다\"라고 명확히 밝힌 뒤 직접/대행 "
        "여부를 물어보세요. 필요 서류를 안내할 때는 법적 명의자(건축주)와 실제 작성 "
        "주체(직접/대행업체 여부에 따라 다름)를 구분해서, 각 서류 옆에 "
        "[본인 준비]/[대행업체 준비]/[관공서 자체 확인]처럼 표시하세요. 판정이 확정되는 "
        "순간 시스템이 이미 관련 법령을 검색해 도구 응답으로 넣어뒀으니 있으면 그대로 "
        "근거로 쓰고, 서류 목록을 안내했으면 record_permit_synthesis(required_documents="
        "[...])로 반드시 기록하세요."
    ),
    4: (
        "\"설계 전에 법적 기준(주차ᆞ소방 등)을 먼저 확인하는 단계입니다 - 여기서 문제가 "
        "나오면 설계를 수정해야 하므로 서류 준비 전에 끝내는 게 좋습니다\"처럼 왜 이 "
        "단계를 하는지 먼저 짚은 뒤, 주차대수ᆞ정화조 용량ᆞ소방시설ᆞ장애인 편의시설ᆞ "
        "위생요구사항 중 실제 해당하는 것만 근거와 함께 안내하세요. 항목 나열로 끝내지 "
        "말고 \"개인이 직접 확인 가능한지 vs 전문가ᆞ관공서 확인이 필요한지\"와 \"실제로 "
        "어떻게 확인하는지\"를 함께 안내하세요. 안내한 항목은 "
        "record_permit_synthesis(pre_diagnosis_items=[...])로 반드시 기록하세요."
    ),
    5: (
        "\"서류가 준비되면 실제로 접수해 허가/신고 수리를 받는 단계입니다\"처럼 짚은 뒤, "
        "관할 시ᆞ군ᆞ구청에 신청ᆞ신고서와 서류를 제출(전자문서ᆞ온라인 접수 가능)하고, "
        "협의가 필요하면 협의기관 검토를 거쳐 수리된다는 흐름을 안내하세요(협의기관이 "
        "확인되면 record_permit_synthesis(related_agencies=[...])). 예상 소요 기간(허가 "
        "15~20일ᆞ신고 3~5일ᆞ기재변경 3~7일 등)을 자연스럽게 덧붙이세요 - "
        "search_regulations로 실제 처리기한을 확인했으면 그 값을 우선 쓰고, 못 찾았으면 "
        "통상치를 참고로만 안내하며 정확한 기간은 관할 구청에 확인하라고 덧붙이세요."
    ),
    6: (
        "이 일을 누가 하는지(건축사무소 / 시공사ᆞ인테리어업체 / 건축주 본인) 반드시 "
        "구분해서 안내하세요. 착공신고 방법(신고서ᆞ설계도서 제출, 관할 시ᆞ군ᆞ구청)을 "
        "안내하세요. 공사감리 필요 여부는 요건이 여러 겹이라 확정 판정하지 말고 검색 "
        "결과의 원칙+예외를 전달한 뒤 \"정확한 판정은 관할 구청에 확인하라\"고 안내하세요. "
        "인테리어ᆞ주방기기ᆞ소방시설 등 설치 시 유의점도 안내하세요(법령 근거 없는 실무 "
        "체크리스트이니 [출처] 없이 그대로 전달)."
    ),
    7: "사용승인 신청 방법ᆞ필요 서류ᆞ준공검사 절차를 안내하세요.",
}


def _actionable_substeps(step):
    """이 Step에서 지금 안내해도 되는(미완료ᆞ비잠금ᆞ해당없음 아니고ᆞinfo도
    아닌) substep만 골라낸다."""
    return [
        sub for sub in step.substeps
        if not (sub.info or sub.done or sub.not_applicable or is_visually_locked(sub))
    ]


def _first_incomplete(steps, lo: int, hi: int):
    """1-based index [lo, hi] 범위에서 처음 "실제로 안내할 미완료 항목이
    남은" Step을 찾는다. Step.done(모든 countable substep이 done이어야 True)은
    안 쓴다 - not_applicable 항목은 countable에 남아있어 영원히 done=False가
    되는 Step이 있는데(예: 신축은 "건축물대장 조회" 항목이 na=True로 항상
    done=False), 그러면 실제로 더 안내할 게 없는 Step에 계속 머물러 다음
    Step으로 못 넘어간다(2026-08-05, current_step_directive 도입 중 발견)."""
    for step in steps:
        if lo <= step.index <= hi and _actionable_substeps(step):
            return step
    return None


def _substep_lines(step) -> list[str]:
    """_actionable_substeps 결과를 텍스트 줄로 만든다. field가 있으면 완료
    확인 시 어떤 도구 인자로 기록해야 하는지 같이 적어둔다."""
    lines = []
    for sub in _actionable_substeps(step):
        ref = f"(사용자가 완료를 확인해주면 record_task_progress({sub.field}=True)로 기록)" if sub.field else ""
        lines.append(f"  - {sub.text} {ref}".rstrip())
    return lines


def current_step_directive(state: "AgentState") -> str | None:
    """건축 트랙(Step1~7)ᆞ창업 트랙(Step8~9)의 "지금 여기" 단계와 그 단계의
    미완료ᆞ비잠금 체크리스트를 계산해 SystemMessage로 얹는다.

    permit 판정이 아직 안 났으면(case_facts._classified가 없으면) Step1~2가
    아직 수집 단계라 반환하지 않는다 - 그 구간은 prompts.py의
    _COLLECT_PERMIT류 지시가 이미 담당한다.
    """
    if not (state.get("case_facts") or {}).get("_classified"):
        return None

    steps = compute_roadmap_steps(state)
    parts: list[str] = []

    construction_step = _first_incomplete(steps, 1, _CONSTRUCTION_TRACK_MAX)
    if construction_step is not None:
        lines = _substep_lines(construction_step)
        if lines:
            block = [f"[현재 단계] Step {construction_step.index}: {construction_step.title}"] + lines
            guidance = STEP_GUIDANCE.get(construction_step.index)
            if guidance:
                block.append(guidance)
            parts.append("\n".join(block))

    startup_step = _first_incomplete(steps, _STARTUP_TRACK_MIN, 9)
    if startup_step is not None:
        lines = _substep_lines(startup_step)
        if lines:
            block = [f"[병행 가능 단계] Step {startup_step.index}: {startup_step.title}"] + lines
            parts.append("\n".join(block))

    if not parts:
        return None

    parts.append(
        "위 목록에서 사용자가 이미 말했거나 완료를 확인해준 항목은 이번 턴에 바로 "
        "record_task_progress로 기록하세요. field 표시가 없는 항목은 이미 다른 기록 "
        "도구(record_case_facts 등)가 담당하는 자동 판정이니 따로 도구를 부르지 마세요. "
        "아직 다루지 않은 항목 중 다음 것부터 자연스럽게 안내하고, 목록에 없는(이미 끝난) "
        "항목은 다시 설명하지 마세요. 여러 항목을 한 번에 나열하지 말고 대화하듯 진행하세요."
    )
    return "\n\n".join(parts)


# 2026-08-05: "간판(Step9) 얘기를 먼저 꺼내면 챗봇이 어떻게 반응해야 하나"는
# 사용자 피드백에서 나온 원칙 - 판정(food/fire/signage)은 2026-07-27 결정대로
# 대화 순서와 무관하게 즉답하되(current_step_directive와 별개로 항상 동작), 그
# 뒤에 "실제로 언제 진행하면 좋은지"(병렬 가능하면 지금 진행 유도, 아니면
# 선행 단계 명시)를 덧붙인다. 잠금 predicate(fire_signage_unlocked/
# food_report_unlocked/business_registration_unlocked)는 src/roadmap_model.py의
# 함수를 그대로 import해서 쓴다(2026-08-05, Phase 3) - 예전엔 이 파일이 같은
# 조건을 로컬 함수로 따로 갖고 있어서 "index.html의 lockReason과 반드시 같은
# 조건이어야 한다"가 주석상의 약속일 뿐이었는데, 이제 둘 다 roadmap_model.py
# 하나만 참조하므로 구조적으로 어긋날 수 없다. 아래 메시지 문구 자체는 여전히
# 채팅 톤에 맞게 따로 쓴 것(STEP_AGENCY_FALLBACK처럼 의도적으로 중복 유지되는
# 표, 2026-08-04 선례와 동일 패턴) - 조건만 단일화했다.
def domain_timing_directive(state: "AgentState") -> str | None:
    """판정이 끝난 도메인(소방ᆞ간판ᆞ식품위생ᆞ위생교육ᆞ사업자등록)에 대해,
    "지금 실제로 진행해도 되는지ᆞ아직 뭘 기다려야 하는지"를 결정론적으로
    계산해 SystemMessage로 얹는다. current_step_directive와 같은 패턴 -
    사실은 여기서 코드가 계산하고, LLM은 그 사실을 옮겨 답변에 반영하기만
    한다(CLAUDE.md 원칙: 판정ᆞ사실 계산은 코드, 설명은 LLM).

    프롬프트가 매 턴 비대해지지 않도록, 결과가 존재하는(판정이 끝난) 도메인만
    골라 한두 줄씩만 담는다 - 판정 자체가 없는 도메인은 아예 언급하지 않는다.
    """
    task_progress = state.get("task_progress") or {}
    food_result = state.get("food_result")
    fire_result = state.get("fire_result")
    signage_result = state.get("signage_result")
    permit_result = state.get("permit_result")

    lines: list[str] = []

    if fire_result is not None or signage_result is not None:
        if fire_signage_unlocked(task_progress):
            lines.append(
                "소방시설ᆞ간판: 공사(시공)가 이미 끝났으니 지금 바로 설치를 "
                "진행하셔도 됩니다."
            )
        else:
            lines.append(
                "소방시설ᆞ간판: 종류ᆞ대상 판정은 지금 확인해드릴 수 있지만, "
                "실제 설치는 공사(시공)가 끝난 뒤에 하시면 됩니다 - 미리 "
                "판정만 확인해두고 시공 마무리 즈음에 진행하시라고 안내하세요."
            )

    if food_result is not None:
        if food_report_unlocked(task_progress):
            lines.append(
                "식품위생 영업신고: 사용승인이 이미 끝났으니 지금 바로 "
                "관할 보건소에 신고 접수를 진행하셔도 됩니다."
            )
        else:
            lines.append(
                "식품위생 영업신고: 어떤 영업 종류인지 판정은 지금 확인해드릴 "
                "수 있지만, 실제 신고 접수는 사용승인이 끝난 뒤에 가능합니다 - "
                "그 전까지는 병렬로 준비만 해두시라고 안내하세요."
            )
        lines.append(
            "위생교육: 사업 진행 단계와 무관하게 언제든 미리 이수하실 수 "
            "있습니다(병렬 진행 가능) - 공사가 진행되는 동안 미리 받아두시길 "
            "권해도 됩니다."
        )

    if permit_result is not None:
        permit_type = getattr(permit_result, "permit_type", None)
        permit_not_needed = permit_type == "인허가불필요"
        if business_registration_unlocked(task_progress, food_result, permit_not_needed):
            lines.append(
                "사업자등록: 임대차계약서 등 서류ᆞ영업신고 종류가 이미 확인됐으니 "
                "지금 바로 관할 세무서에 신청하셔도 됩니다(공사 완료를 기다릴 "
                "필요 없음, 병렬 진행 가능)."
            )
        elif food_result is not None:
            lines.append(
                "사업자등록: 임대차계약서 등 서류ᆞ영업신고 종류 확인이 끝나야 "
                "진행할 수 있습니다 - 공사(시공) 완료를 기다릴 필요는 없고, "
                "그 두 가지만 준비되면 공사 중이라도 바로 신청 가능하다고 "
                "안내하세요."
            )

    if not lines:
        return None
    return "[실행 시점 참고 - 사용자가 순서를 벗어나 물어봐도 이 사실 그대로 반영] " + " ".join(lines)
