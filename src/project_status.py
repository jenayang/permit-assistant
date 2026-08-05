"""Project Status - 창업 로드맵(Step 0~4)의 구조화된 진행 상태를 계산.

roadmap.py의 current_step_directive()가 LLM 프롬프트에 넣을 "지금 여기 단계"
텍스트를 만드는 것과는 별개다. 이 모듈은 프론트엔드 여러 화면(재접속 요약
배너, 사이드바 Next Action 위젯, 향후 모바일 카드 등)이 공유해서 쓸 구조화된
데이터를 계산한다 - "같은 데이터를 다르게 보여주는 문제"라는 원칙(2026-07-28
논의)에 따라, progress/completed/트랙별 상태/summary를 이 함수 한 곳에서만
계산하고 화면들은 그 결과만 갖다 쓴다(직접 상태를 재해석하지 않는다).

완료 판정 기준은 src/roadmap_model.py의 compute_roadmap_steps()가 계산한
Step1~9 결과에서 그대로 가져온다(2026-08-05, Phase 2 - 예전엔 이 파일이
case_facts/permit_result/... 등에서 직접 자체적으로 판정 bool을 재계산해서
src/static/index.html의 판정과 어긋날 위험이 있었다. 이제 둘 다
roadmap_model.py 하나만 참조하므로 구조적으로 어긋날 수 없다).

이 파일이 쓰는 5그룹(STEP_LABELS)은 Step1~9보다 훨씬 거친 축약이다 - 예를
들어 "건축 인허가" 한 그룹이 실제로는 Step2~5(신고ᆞ허가 판단ᆞ건축사
선정ᆞ사전검토ᆞ신청접수)를 합친 것이다. 그리고 이 5그룹의 15개 항목은
Step1~9의 모든 substep(약 21개, info 참고용 제외)을 다 세지 않고 그중
task_progress 필드가 있는 항목(사용자가 실제로 체크하는 것)만 골라 쓴다 -
Step1의 "사업 예정지 주소 확인"처럼 아직 어떤 서버 도구도 값을 채우지 않는
substep(알려진 공백, roadmap_model.py 모듈 docstring 참고)을 그대로
분모에 넣으면 그 항목이 영원히 못 채워져 progress%가 100%에 절대 도달 못하는
회귀가 생긴다 - 그래서 이 파일은 기존에 실제로 쓰던 15개 항목 구성을
그대로 유지한 채(라벨ᆞ개수 불변), "그 항목들의 완료 여부를 어디서 읽어오는지"
만 roadmap_model.py의 단일 소스로 바꿨다.

트랙 분리(2026-07-28 수정): 처음엔 Step 0~4를 하나의 순서로 훑어서
current_step 하나만 계산했는데, 실사용 세션(79430043)에서 문제가 드러났다 -
건축 트랙(Step0~2)이 안 끝난 채로 창업 준비 트랙(Step3~4, 식품위생 등)이
먼저 진행되면(도메인이 실제로 병렬이라 흔한 일), 위젯은 계속 "Step1"이라고만
말해서 사용자가 지금 위생교육 얘기를 하고 있는데도 진행 상황이 안 맞는
것처럼 보였다. roadmap.py의 current_step_directive()ᆞ프론트 로드맵(시안 A2)은
이미 "건축 트랙(순차)ᆞ창업 준비 트랙(병렬)"으로 나눠서 다루고 있어서, 이
함수도 같은 기준으로 트랙별 current_step/next_action을 따로 계산하도록
맞췄다 - 단일 current_step 필드는 없앴다(두 트랙 상태를 하나로 뭉개면 다시
같은 문제가 재발함).
"""
from __future__ import annotations

from src.roadmap_model import Step, compute_roadmap_steps, roadmap_steps_to_dicts

STEP_LABELS = ["건축 유형 확인", "건축 인허가", "공사", "창업 행정", "오픈 준비"]

# 건축 트랙(Step 0~2, 실제로 순서가 있는 절차)과 창업 준비 트랙(Step 3~4,
# 인허가 트랙과 무관하게 병렬 진행 가능) - roadmap.py의 current_step_directive()
# 와 동일한 구분 기준.
_CONSTRUCTION_TRACK_INDICES = (0, 1, 2)
_STARTUP_TRACK_INDICES = (3, 4)


def _step_substeps(roadmap_steps: list[Step]) -> list[list[tuple[str, bool]]]:
    """Step 0~4(5그룹) 각각의 [(라벨, 완료여부), ...] - compute_roadmap_steps()가
    이미 계산한 Step1~9의 substep을 위치로 찾아 done을 그대로 재사용한다(중복
    판정 없음). 각 튜플 옆 주석은 roadmap_steps에서의 출처(1-based Step
    번호ᆞ0-based substep 인덱스) - roadmap_model.py의 substep 순서가 바뀌면
    이 위치 참조도 같이 갱신해야 한다(tests/test_project_status.py가 어긋나면
    바로 드러남)."""
    s = roadmap_steps

    def done_or_na(step_idx: int, sub_idx: int) -> bool:
        # 인테리어 전용(interior_only) 공사는 착공신고ᆞ시공 substep 자체가
        # notApplicable로 닫히고 done은 영원히 False로 남는다(roadmap_model.py
        # 모듈 docstring 참고) - "완료됐거나 애초에 해당 없음"을 완료로 친다.
        sub = s[step_idx].substeps[sub_idx]
        return sub.done or sub.not_applicable

    return [
        [
            ("행위 유형 확인", s[0].substeps[1].done),  # Step1 substep[1]
            ("허가ᆞ신고ᆞ기재변경 대상 판정", s[1].substeps[0].done),  # Step2 substep[0]
        ],
        [
            ("사전 검토", s[3].substeps[-1].done),  # Step4 마지막 substep(사전 검토 확인 완료)
            ("설계ᆞ서류 준비", s[2].substeps[-1].done),  # Step3 마지막 substep(설계ᆞ서류 준비 완료 확인)
            ("신청ᆞ접수", s[4].substeps[6].done),  # Step5 substep[6](관할 구청 방문ᆞ접수)
        ],
        [
            ("착공신고", done_or_na(5, 1)),  # Step6 substep[1](착공신고 완료 확인)
            ("공사(시공)", done_or_na(5, 5)),  # Step6 substep[5](시공 완료 확인)
            ("사용승인", s[6].substeps[5].done),  # Step7 substep[5](사용승인 완료 확인)
        ],
        [
            ("식품위생 영업신고 확인", s[7].substeps[0].done),  # Step8 substep[0]
            ("간판ᆞ옥외광고물 확인", s[8].substeps[0].done),  # Step9 substep[0]
            ("사업자등록", s[7].substeps[3].done),  # Step8 substep[3]
            ("위생교육 이수", s[7].substeps[4].done),  # Step8 substep[4]
        ],
        [
            ("인테리어ᆞ장비 설치", s[5].substeps[6].done),  # Step6 substep[6]
            # 직원을 안 두는 1인ᆞ가족 운영 사업자는 4대보험 가입 신고 자체가
            # 대상이 아니다 - roadmap_model.py의 이 substep은 이미
            # hires_staff=False 우회를 done에 반영해 계산한다.
            ("직원 등록(4대보험 가입)", s[8].substeps[2].done),  # Step9 substep[2]
            ("영업 시작", s[8].substeps[8].done),  # Step9 substep[8]
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
         "summary": str, "requires_architect": bool | None,
         "steps": list[dict]}  # Step1~9 상세(camelCase, roadmap_model.Step 참고)
    """
    roadmap_steps = compute_roadmap_steps(state)
    steps = _step_substeps(roadmap_steps)

    construction = _track_status(steps, _CONSTRUCTION_TRACK_INDICES)
    startup = _track_status(steps, _STARTUP_TRACK_INDICES)
    completed = construction["completed"] + startup["completed"]

    total_done = sum(sum(1 for _, done in s if done) for s in steps)
    total_all = sum(len(s) for s in steps)
    progress = round(total_done / total_all * 100) if total_all else 0

    # 2026-08-05: "다음 할 일" 위젯이 건축 트랙이 막 시작한 시점에도 계속
    # "창업 준비 트랙: 식품위생 영업신고 확인"을 먼저 알려줘서 혼란을 준다는
    # 신고 - startup_track의 current_step/next_action은 건축 트랙이 "공사"
    # 단계에 진입한 뒤(또는 건축 트랙 자체가 이미 끝난 뒤)에만 노출한다.
    # completed 집계ᆞ전체 progress%는 그대로 실제 완료 사실을 반영해야 하니
    # 건드리지 않는다 - 여기서 숨기는 건 "먼저 권하는 다음 행동" 노출뿐.
    construction_reached_construction_phase = construction["current_step"] in ("공사", None)
    startup_current_step = startup["current_step"] if construction_reached_construction_phase else None
    startup_next_action = startup["next_action"] if construction_reached_construction_phase else None

    # "다음 하셔야 할 일"을 먼저 말하도록 재구성(2026-08-05 피드백: "로드맵
    # 결과 단계로 들어오자마자 챗봇이 사용자가 해야할 일을 바로 알려줘 -
    # 현재 안내는 현재 어느 단계인지 확인하는 역할만 함"). 예전엔 "현재 OO
    # 단계입니다"처럼 상태 리캡이 먼저였는데, 지금 뭘 해야 하는지(다음 할
    # 일)를 문장 맨 앞으로 옮기고 지난 완료 내역은 참고로 뒤에 붙인다.
    action_phrases = []
    if construction["current_step"] and construction["next_action"]:
        action_phrases.append(f"건축: {construction['next_action']}")
    if startup_current_step and startup_next_action:
        action_phrases.append(f"창업 준비: {startup_next_action}")

    if total_done == 0:
        summary = "아직 시작 전이에요. 창업 준비 중인 업종ᆞ장소를 알려주시면 진행 상황을 기록해드릴게요."
    elif not action_phrases:
        summary = "모든 절차를 완료하셨어요! 🎉"
    else:
        completed_note = f" (지금까지 {', '.join(completed)} 완료, 진행률 {progress}%)" if completed else ""
        summary = f"지금 하셔야 할 일은 {' · '.join(action_phrases)}입니다.{completed_note}"

    # 건축법 제23조ᆞ제19조6항: 이 케이스가 건축사사무소 설계 의무 대상인지.
    # True=대행 필요, False=개인 직접 가능, None=정보 부족. 판정은 규칙 엔진이
    # 하고(permit.py) roadmap_steps[2](Step3ᆞ건축사사무소 선정)가 이미 그
    # 결과를 실어뒀으니 여기서 다시 계산하지 않고 그대로 재사용한다(단일 소스).
    requires_architect = roadmap_steps[2].architect

    return {
        "progress": progress,
        "completed": completed,
        "construction_track": {"current_step": construction["current_step"], "next_action": construction["next_action"]},
        "startup_track": {"current_step": startup_current_step, "next_action": startup_next_action},
        "summary": summary,
        "requires_architect": requires_architect,
        # Step1~9 상세 체크리스트(2026-08-05, Phase 4) - camelCase dict라
        # /project-status(StepOut 스키마)ᆞ/query(project_status: dict 그대로
        # 통과)ᆞ/facts 세 응답 경로 모두 같은 모양으로 나간다.
        "steps": roadmap_steps_to_dicts(roadmap_steps),
    }
