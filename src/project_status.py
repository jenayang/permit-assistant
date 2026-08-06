"""Project Status - 창업 로드맵(Step 0~4)의 구조화된 진행 상태를 계산.

agent.py의 _roadmap_status_summary()가 LLM 프롬프트에 넣을 "순차/병렬 트랙"
텍스트를 만드는 것과는 별개다. 이 모듈은 프론트엔드 여러 화면(재접속 요약
배너, 사이드바 Next Action 위젯, 향후 모바일 카드 등)이 공유해서 쓸 구조화된
데이터를 계산한다 - "같은 데이터를 다르게 보여주는 문제"라는 원칙(2026-07-28
논의)에 따라, progress/completed/트랙별 상태/summary를 이 함수 한 곳에서만
계산하고 화면들은 그 결과만 갖다 쓴다(직접 상태를 재해석하지 않는다).

완료 판정 기준은 src/static/index.html의 renderRoadmap()/buildChecklists()가
쓰는 기준과 반드시 같아야 한다(판정 도메인은 case_facts/permit_result/
food_result/fire_result/signage_result, task_progress 도메인은 사용자
자기보고) - 안 그러면 로드맵 UI와 이 API가 서로 다른 진행률을 말하게 된다.

트랙 분리(2026-07-28 수정): 처음엔 Step 0~4를 하나의 순서로 훑어서
current_step 하나만 계산했는데, 실사용 세션(79430043)에서 문제가 드러났다 -
건축 트랙(Step0~2)이 안 끝난 채로 창업 준비 트랙(Step3~4, 식품위생 등)이
먼저 진행되면(도메인이 실제로 병렬이라 흔한 일), 위젯은 계속 "Step1"이라고만
말해서 사용자가 지금 위생교육 얘기를 하고 있는데도 진행 상황이 안 맞는
것처럼 보였다. agent.py의 _roadmap_status_summary()ᆞ프론트 로드맵(시안 A2)은
이미 "건축 트랙(순차)ᆞ창업 준비 트랙(병렬)"으로 나눠서 다루고 있어서, 이
함수도 같은 기준으로 트랙별 current_step/next_action을 따로 계산하도록
맞췄다 - 단일 current_step 필드는 없앴다(두 트랙 상태를 하나로 뭉개면 다시
같은 문제가 재발함).
"""
from __future__ import annotations

from src.agents.permit import requires_licensed_architect

STEP_LABELS = ["건축 유형 확인", "건축 인허가", "공사", "창업 행정", "오픈 준비"]

# 건축 트랙(Step 0~2, 실제로 순서가 있는 절차)과 창업 준비 트랙(Step 3~4,
# 인허가 트랙과 무관하게 병렬 진행 가능) - agent.py의 _roadmap_status_summary()
# 와 동일한 구분 기준.
_CONSTRUCTION_TRACK_INDICES = (0, 1, 2)
_STARTUP_TRACK_INDICES = (3, 4)


def _step_substeps(state: dict) -> list[list[tuple[str, bool]]]:
    """Step 0~4 각각의 [(라벨, 완료여부), ...]."""
    case_facts = state.get("case_facts") or {}
    permit_result = state.get("permit_result")
    food_result = state.get("food_result")
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
            # 직원을 안 두는 1인ᆞ가족 운영 사업자는 4대보험 가입 신고 자체가
            # 대상이 아니다 - hires_staff=False로 명시되면 이 항목이 영원히
            # 미완료로 남아 진행률이 100%를 못 채우는 문제가 있었다
            # (2026-07-28, 세션 2cda4d67류에서 신고).
            ("직원 등록(4대보험 가입)", bool(tp.get("staff_registration")) or tp.get("hires_staff") is False),
            ("영업 시작", bool(tp.get("opened"))),
        ],
    ]


def _track_status(steps: list[list[tuple[str, bool]]], indices: tuple[int, ...]) -> dict:
    """트랙(Step 인덱스 목록) 하나의 진행 상태 - 그 트랙 안에서만 순서대로
    훑어 첫 미완료 단계를 current_step으로 잡는다(트랙 간에는 서로 훑지
    않는다 - 이게 이번에 고친 핵심)."""
    current_step = None
    next_action = None
    completed = []
    for i in indices:
        label, substeps = STEP_LABELS[i], steps[i]
        done_count = sum(1 for _, done in substeps if done)
        if done_count == len(substeps):
            completed.append(label)
        elif current_step is None:
            current_step = label
            next_action = next(text for text, done in substeps if not done)
    return {"current_step": current_step, "next_action": next_action, "completed": completed}


def compute_project_status(state: dict) -> dict:
    """상태(case_facts/permit_result/.../task_progress)에서 프로젝트 진행
    현황을 계산한다. LLM을 부르지 않는 순수 결정론적 함수(규칙 기반) -
    summary 문장도 템플릿으로만 조립해서 지어낼 여지가 없다.

    Returns:
        {"progress": int, "completed": list[str],
         "construction_track": {"current_step", "next_action"},
         "startup_track": {"current_step", "next_action"},
         "summary": str}
    """
    steps = _step_substeps(state)

    construction = _track_status(steps, _CONSTRUCTION_TRACK_INDICES)
    startup = _track_status(steps, _STARTUP_TRACK_INDICES)
    completed = construction["completed"] + startup["completed"]

    total_done = sum(sum(1 for _, done in s if done) for s in steps)
    total_all = sum(len(s) for s in steps)
    progress = round(total_done / total_all * 100) if total_all else 0

    # 창업 준비 트랙(병렬) "지금 할 일" 노출은 공사(Step2)가 실제로 시작된
    # 뒤부터만 - agent.py/roadmap.py의 _roadmap_status_summary와 정확히 같은
    # 이유(공사 전엔 병렬 진행을 권할 근거가 없음, 2026-08-06 재신고: 이
    # 함수가 재접속 요약 배너ᆞ상단 위젯에 항상 창업 행정을 같이 알려주고
    # 있어서 챗봇 쪽 게이트와 무관하게 새는 걸 뒤늦게 발견). 이미 완료된
    # 항목(completed/progress)은 실제로 한 일이라 그대로 인정하고, "다음
    # 할 일"만 미룬다.
    tp = state.get("task_progress") or {}
    step2_started = any(tp.get(f) for f in ("construction_notice", "construction")) or bool(tp.get("use_approval"))
    if not step2_started:
        startup = {**startup, "current_step": None, "next_action": None}

    track_phrases = []
    if construction["current_step"]:
        track_phrases.append(f"건축 트랙은 '{construction['current_step']}' 단계(다음 할 일: '{construction['next_action']}')")
    if startup["current_step"]:
        track_phrases.append(f"창업 준비 트랙은 '{startup['current_step']}' 단계(다음 할 일: '{startup['next_action']}')")

    if total_done == 0:
        summary = "아직 시작 전이에요. 창업 준비 중인 업종ᆞ장소를 알려주시면 진행 상황을 기록해드릴게요."
    elif not track_phrases:
        summary = "모든 절차를 완료하셨어요! 🎉"
    elif not completed:
        summary = "현재 " + ", ".join(track_phrases) + "입니다."
    else:
        summary = (
            f"지난번에는 {', '.join(completed)}까지 완료하셨어요(진행률 {progress}%). "
            + ", ".join(track_phrases) + "입니다."
        )

    # 건축법 제23조ᆞ제19조6항: 이 케이스가 건축사사무소 설계 의무 대상인지.
    # True=대행 필요, False=개인 직접 가능, None=정보 부족. 판정은 규칙 엔진이
    # 하고(permit.py) 여기선 그 결과만 실어 내려보낸다 - 프론트가 JS로 다시
    # 계산하지 않게(단일 소스).
    requires_architect = requires_licensed_architect(state.get("case_facts") or {})

    return {
        "progress": progress,
        "completed": completed,
        "construction_track": {"current_step": construction["current_step"], "next_action": construction["next_action"]},
        "startup_track": {"current_step": startup["current_step"], "next_action": startup["next_action"]},
        "summary": summary,
        "requires_architect": requires_architect,
    }
