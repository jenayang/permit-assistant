"""Roadmap Progress - 사용자가 실제로 완료했다고 밝힌 창업 절차 항목 추적.

permit.py/food_safety.py 등(규칙 엔진이 "뭘 해야 하는지" 계산)과는 성격이
다르다. 여기서 다루는 항목(착공신고ᆞ시공ᆞ사용승인ᆞ사업자등록ᆞ위생교육ᆞ
인테리어ᆞ장비설치ᆞ직원등록ᆞ영업개시)은 전부 실물 세계에서 사용자가
직접 하는 일이라, AI가 스스로 판단하거나 다른 도구로 확인할 방법이 없다 -
오직 사용자가 명시적으로 "완료했다"고 말한 것만 기록한다.

"안내를 보여줬다"와 "완료했다"를 혼동하지 않는 게 이 모듈의 핵심 원칙이다
(2026-07-27 논의) - 예를 들어 get_business_registration_guide 도구가
호출됐다고 해서 사업자등록이 끝난 게 아니다. 그래서 이 상태는 record_*_facts
류(판정에 쓰이는 사실)와도 다르고, get_*_guide류(정적 안내)와도 별개의
채널(task_progress)로 관리된다.

record_case_facts 등과 동일하게 문자열 키를 자유롭게 받는 범용 함수
(예: record_progress(step="x", status="y"))가 아니라 타입이 고정된 개별
파라미터로 설계했다 - 자유 문자열 키는 LLM이 오타를 내거나 임의의 키를
만들어낼 위험이 있어, 이 프로젝트 전반의 컨벤션(고정 스키마)과도 안 맞는다.
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# record_task_progress의 파라미터 이름과 반드시 일치해야 한다 - 프론트 로드맵
# 체크박스(사용자가 직접 클릭해서 완료 표시)가 /task-progress API로 이 목록에
# 있는 필드만 갱신하도록 api.py가 검증할 때 재사용한다. 여기 하나만 고치면
# 도구ᆞAPI 양쪽에 다 반영되도록 단일 소스로 둔다.
TASK_PROGRESS_FIELDS: list[str] = [
    "pre_diagnosis_checked", "documents_prepared", "application_submitted",
    "construction_notice", "construction", "use_approval",
    "business_registration", "hygiene_education",
    "interior_equipment", "staff_registration", "opened",
    "hires_staff", "uses_agency", "architect_selected", "interior_only",
]


@tool
def record_task_progress(
    pre_diagnosis_checked: bool | None = None,
    documents_prepared: bool | None = None,
    application_submitted: bool | None = None,
    construction_notice: bool | None = None,
    construction: bool | None = None,
    use_approval: bool | None = None,
    business_registration: bool | None = None,
    hygiene_education: bool | None = None,
    interior_equipment: bool | None = None,
    staff_registration: bool | None = None,
    opened: bool | None = None,
    hires_staff: bool | None = None,
    uses_agency: bool | None = None,
    architect_selected: bool | None = None,
    interior_only: bool | None = None,
) -> str:
    """사용자가 실제로 완료했다고 명시적으로 말한 창업 절차 항목을 기록하세요.

    - **사용자가 "~했어요", "~끝났어요", "~마쳤어요"처럼 완료를 명확히 밝힌
      경우에만 True로 호출하세요.** "~하려고요", "~할 예정이에요", "~해야
      하나요?"처럼 계획ᆞ의도ᆞ질문 표현은 절대 기록하지 마세요 - 완료와
      계획을 혼동하면 진행률이 거짓 정보가 됩니다.
    - 이 항목들은 실제 세상에서 벌어지는 일(착공ᆞ사업자등록ᆞ영업개시 등)이라
      AI가 스스로 판단하거나 안내를 보여줬다는 이유로 완료 처리하면 안
      됩니다 - 반드시 사용자가 직접 완료를 말해준 경우에만 기록하세요.
      **특히 pre_diagnosis_checked/documents_prepared는 record_permit_synthesis로
      사전진단ᆞ서류 항목을 "설명"한 것과 다릅니다** - 설명은 안내일 뿐이고,
      사용자가 "사전 진단 항목 확인했어요ᆞ문제없어요"/"서류 다 준비했어요"처럼
      직접 확인해줬을 때만 기록하세요. application_submitted도 마찬가지로
      [Step 1-4] 안내를 받았다는 것과 무관하게, 사용자가 "접수했어요"/
      "신청서 냈어요"처럼 실제 접수를 직접 말해줬을 때만 기록하세요.
    - 이번 턴에 새로 알게 된 항목만 채우고 나머지는 생략(None)하세요. 여러
      턴에 걸쳐 나눠서 호출해도 이전에 기록한 값 위에 누적됩니다.
    - False로 기록할 필요는 거의 없습니다(언급이 없으면 자동으로 미완료
      상태) - 사용자가 "아직 안 했어요"처럼 명시적으로 되돌리는 경우에만
      False를 쓰세요.
    - **직원을 안 두고 혼자ᆞ가족끼리만 운영한다고 명확히 밝히면
      hires_staff=False로 기록하세요** - 4대보험 가입 신고는 직원을 채용할
      때만 필요한 절차라, 직원이 없으면 staff_registration 자체가 대상이
      아닙니다(그렇지 않으면 진행률이 100%를 못 채웁니다). 나중에 직원을
      채용하겠다고 하면 hires_staff=True로 다시 바꾸세요.
    - **건축사사무소 선정이 의무가 아닐 때 "직접 하겠다/도움받겠다"를 명확히 밝히면
      uses_agency로 기록하세요** - 이후 사전검토ᆞ서류 준비 안내가 이 값에 따라
      갈립니다. 실제로 사무소ᆞ업체를 정했다고 밝히면 architect_selected=True로
      기록하세요.
    - **구조 공사 없이 인테리어만 진행한다고 명확히 밝히면 interior_only=True로
      기록하세요** - 착공신고ᆞ시공 항목 자체가 대상이 아니게 됩니다.

    Args:
        pre_diagnosis_checked: 사전 진단 항목(주차ᆞ정화조ᆞ소방시설 등) 확인을 완료했는지
        documents_prepared: 건축 인허가 필수 서류 준비를 완료했는지
        application_submitted: 관할 구청에 신청서ᆞ신고서 접수를 완료했는지
        construction_notice: 착공신고를 완료했는지
        construction: 착공ᆞ시공을 완료했는지(또는 진행 중이라고 밝혔는지)
        use_approval: 사용승인을 받았는지
        business_registration: 사업자등록을 완료했는지
        hygiene_education: 위생교육 이수를 완료했는지
        interior_equipment: 인테리어ᆞ장비 설치를 완료했는지
        staff_registration: 직원 등록(4대보험 가입)을 완료했는지
        opened: 실제로 영업을 시작했는지
        hires_staff: 직원을 채용하는지(False면 직원 등록 항목이 대상 제외로 처리됨)
        uses_agency: 건축사사무소 설계가 법적 의무가 아닐 때, 대행업체(건축사사무소ᆞ행정사)
            도움을 받는지(True) 직접 진행하는지(False)
        architect_selected: 건축사사무소(또는 대행업체)를 실제로 선정 완료했는지
        interior_only: 구조 공사(신축ᆞ증축ᆞ대수선 등) 없이 인테리어만 진행하는지
    """
    logger.info("[도구] record_task_progress(%r)", {
        k: v for k, v in locals().items() if v is not None
    })
    return "기록됨."


# 사용승인은 법적으로 착공신고ᆞ시공이 끝나야만 가능한 절차라, 뒷단계가
# True로 기록되면 앞단계도 자동으로 True 채워도 안전하다(2026-08-04 -
# "사용승인 완료했는데 착공신고ᆞ시공 체크박스가 그대로"라는 신고에 대응).
_CONSTRUCTION_ORDER = ("construction_notice", "construction", "use_approval")


def cascade_construction_progress(update: dict) -> dict:
    """update에 뒤 단계가 True로 있으면 앞 단계도(명시값 없으면) True로 채운다.

    이미 명시적으로 다른 값(예: False)이 들어있는 필드는 덮어쓰지 않는다 -
    setdefault라 update에 없던 키만 채워진다."""
    result = dict(update)
    for i in range(len(_CONSTRUCTION_ORDER) - 1, 0, -1):
        if result.get(_CONSTRUCTION_ORDER[i]) is True:
            for earlier in _CONSTRUCTION_ORDER[:i]:
                result.setdefault(earlier, True)
    return result
