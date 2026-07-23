"""Permit Agent - 허가/신고/기재변경 판정(규칙 기반) + 절차 로드맵.

설계서(4번 섹션)는 "Regulation Agent가 검색한 법령을 LLM이 보고 판단"하는
구조를 제안하지만, 실제로는 허가/신고 여부가 법령상 객관적 기준(면적ᆞ층수ᆞ
시설군)으로 정해지는 값이라 LLM 판단에 맡기지 않고 classify_case()가 파이썬
규칙으로 결정론적으로 계산한다(LLM이 도구 호출을 빼먹거나 기준을 잘못
계산하는 문제를 실제로 겪은 뒤 도입 - agent.py 참고). Regulation Agent가
검색한 법령은 최종 설명(explanation) 문구를 만들 때만 LLM이 참고한다.
"""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.agents.legal_data import get_thresholds

logger = logging.getLogger(__name__)


# 절차 결과(허가/신고/기재변경 등)별 로드맵. 예전엔 "케이스 유형(신축/용도변경/대수선)"
# 기준으로 트리를 나누고 허가·신고 판정 자체를 사용자에게 되묻는 분기로 넣었었는데,
# 그 판정은 법령상 규칙 기반으로 결정되는 부분이라 사용자가 알 수 없는 걸 물어보는
# 셈이었다(예: "허가 대상인가요 신고 대상인가요?"). 이제는 classify_case()가 파이썬
# 로직으로 허가/신고/기재변경을 미리 계산해서 결과 트리의 root로 바로 진입시키고,
# 트리에는 사용자만 답할 수 있는 분기(소유자/임차인)만 남긴다.
#
# mermaid 다이어그램은 이 dict에서 generate_mermaid()로 생성하므로, 트리를
# 고치면 그림도 같이 최신 상태로 유지된다 - 손으로 안 맞춰도 됨.
#
# 노드 형태:
#   분기: {"type": "branch", "question": "...", "options": {"선택지": "다음노드id"}}
#   단계: {"type": "stage", "label": "...", "next": "다음노드id" | None(=완료/종료)}
PROCEDURE_TREE: dict[str, dict] = {
    "건축허가": {
        "root": "owner_check",
        "nodes": {
            "owner_check": {
                "type": "branch", "question": "소유자이신가요, 임차인이신가요?",
                "options": {"소유자": "apply_owner", "임차인": "apply_tenant"},
            },
            "apply_owner": {"type": "stage", "label": "허가 신청 (소유권 증빙)", "next": "construction_notice"},
            "apply_tenant": {"type": "stage", "label": "허가 신청 (토지사용승낙서 등)", "next": "construction_notice"},
            "construction_notice": {"type": "stage", "label": "착공신고", "next": "construction"},
            "construction": {"type": "stage", "label": "착공/시공", "next": "approval"},
            "approval": {"type": "stage", "label": "사용승인", "next": "done"},
            "done": {"type": "stage", "label": "완료", "next": None},
        },
    },
    "건축신고": {
        "root": "owner_check",
        "nodes": {
            "owner_check": {
                "type": "branch", "question": "소유자이신가요, 임차인이신가요?",
                "options": {"소유자": "apply_owner", "임차인": "apply_tenant"},
            },
            "apply_owner": {"type": "stage", "label": "신고 (소유권 증빙)", "next": "construction"},
            "apply_tenant": {"type": "stage", "label": "신고 (토지사용승낙서 등)", "next": "construction"},
            "construction": {"type": "stage", "label": "착공/시공", "next": "approval"},
            "approval": {"type": "stage", "label": "사용승인", "next": "done"},
            "done": {"type": "stage", "label": "완료", "next": None},
        },
    },
    "용도변경허가": {
        "root": "owner_check",
        "nodes": {
            "owner_check": {
                "type": "branch", "question": "소유자이신가요, 임차인이신가요?",
                "options": {"소유자": "docs_owner", "임차인": "docs_tenant"},
            },
            "docs_owner": {"type": "stage", "label": "용도변경 허가 신청 (소유권 증빙)", "next": "done"},
            "docs_tenant": {"type": "stage", "label": "용도변경 허가 신청 (임대차계약서 + 소유자 동의서)", "next": "done"},
            "done": {"type": "stage", "label": "완료(용도변경 반영)", "next": None},
        },
    },
    "용도변경신고": {
        "root": "owner_check",
        "nodes": {
            "owner_check": {
                "type": "branch", "question": "소유자이신가요, 임차인이신가요?",
                "options": {"소유자": "docs_owner", "임차인": "docs_tenant"},
            },
            "docs_owner": {"type": "stage", "label": "용도변경 신고 (소유권 증빙)", "next": "done"},
            "docs_tenant": {"type": "stage", "label": "용도변경 신고 (임대차계약서 + 소유자 동의서)", "next": "done"},
            "done": {"type": "stage", "label": "완료(용도변경 반영)", "next": None},
        },
    },
    "건축물대장기재변경": {
        "root": "apply",
        "nodes": {
            "apply": {"type": "stage", "label": "건축물대장 기재사항 변경 신청", "next": "done"},
            "done": {"type": "stage", "label": "완료", "next": None},
        },
    },
    "가설건축물허가": {
        "root": "apply",
        "nodes": {
            "apply": {"type": "stage", "label": "가설건축물 축조허가 신청", "next": "construction"},
            "construction": {"type": "stage", "label": "축조", "next": "manage"},
            "manage": {"type": "stage", "label": "존치기간 관리", "next": None},
        },
    },
    "가설건축물신고": {
        "root": "apply",
        "nodes": {
            "apply": {"type": "stage", "label": "가설건축물 축조신고", "next": "construction"},
            "construction": {"type": "stage", "label": "축조", "next": "manage"},
            "manage": {"type": "stage", "label": "존치기간 관리", "next": None},
        },
    },
    "인허가불필요": {
        "root": "done",
        "nodes": {
            "done": {"type": "stage", "label": "별도 인허가 없이 진행 가능", "next": None},
        },
    },
}


# --- 규칙 기반 분류 (classify_case) --------------------------------------
# 사용자 상황(case_facts)을 받아 위 PROCEDURE_TREE의 어느 결과(허가/신고/기재변경 등)에
# 해당하는지 "결정론적으로" 계산한다. LLM 판단에 안 맡기는 이유: 이 값들은 법령에 명시된
# 객관적 기준(면적/층수/시설군 등)이라 애매함이 없고, LLM이 매번 정확히 계산해줄 거라고
# 신뢰할 수 없다(오늘 실제로 도구 호출을 빼먹는 문제를 겪음). 판단 흐름(if/elif 분기
# 구조)은 여기 파이썬 코드가 갖고 있지만, 그 흐름이 참조하는 숫자ᆞ집합 값은
# permit_thresholds.yaml(get_thresholds())에서 가져온다 - 법령 개정으로 숫자만
# 바뀌는 경우(예: 85㎡→90㎡) 이 파일을 안 건드리고 YAML 값만 고치면 되도록 분리한
# 것. 다만 아예 새로운 조건/분기가 생기는 개정이면 이 if/elif 구조 자체를 고쳐야
# 한다 - YAML 분리가 그런 구조 변경까지 없애주지는 않는다. 원문 대조 완료(2026-07-22 세션).
#
# 각 act_type별로 classify_case가 요구하는 case_facts 필드. 하나라도 비어있으면(None)
# 아직 분류 안 함(None 반환) — 그래프가 이걸로 "정보 충분?" 판단.
REQUIRED_FIELDS: dict[str, list[str]] = {
    "신축": ["size_sqm", "floors", "land_zone"],
    "증축": ["extension_size_sqm"],
    "개축": ["extension_size_sqm"],
    "재축": ["extension_size_sqm"],
    "이전": [],
    "대수선": ["renovation_scope", "size_sqm", "floors"],
    "용도변경": ["current_facility_group", "desired_facility_group"],
    "일반수선": [],
    "가설건축물": ["temporary_purpose", "temporary_duration_years", "temporary_is_concrete"],
}

# 시설군 9개 + 소속 세부 용도 (건축법 시행령 제14조제5항, 원문 대조 완료).
# 번호가 작을수록 상위군. LLM이 "업무시설→카페" 같은 서술을 시설군 번호로
# 직접 매핑할 수 있도록 세부 항목까지 record_case_facts 설명에 넣어준다
# (안 그러면 무엇을 물어봐야 할지 판단을 못 해서 엉뚱한 걸 되물음 - 실제로 겪음).
FACILITY_GROUPS = {
    1: ("자동차 관련 시설군", ["자동차 관련 시설"]),
    2: ("산업 등 시설군", ["운수시설", "창고시설", "공장", "위험물저장 및 처리시설", "자원순환 관련 시설", "묘지 관련 시설", "장례시설"]),
    3: ("전기통신시설군", ["방송통신시설", "발전시설"]),
    4: ("문화집회시설군", ["문화 및 집회시설", "종교시설", "위락시설", "관광휴게시설"]),
    5: ("영업시설군", ["판매시설", "운동시설", "숙박시설", "제2종 근린생활시설 중 다중생활시설"]),
    6: ("교육 및 복지시설군", ["의료시설", "교육연구시설", "노유자시설", "수련시설", "야영장 시설"]),
    7: ("근린생활시설군", ["제1종 근린생활시설", "제2종 근린생활시설(다중생활시설 제외, 카페ᆞ음식점 등 대부분 포함)"]),
    8: ("주거업무시설군", ["단독주택", "공동주택", "업무시설(사무실 등)", "교정시설", "국방ᆞ군사시설"]),
    9: ("그 밖의 시설군", ["동물 및 식물 관련 시설"]),
}


def _missing_fields(act_type: str, facts: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS.get(act_type, []) if facts.get(f) is None]


def classify_case(facts: dict) -> str | None:
    """case_facts로 PROCEDURE_TREE 최상위 키(결과)를 결정. 정보 부족하면 None.

    facts에서 쓰는 키: act_type, size_sqm(연면적, ㎡), floors(층수),
    extension_size_sqm(증축·개축·재축 대상 부분 바닥면적, ㎡), land_zone
    ("관리지역"|"농림지역"|"자연환경보전지역"|"기타"), renovation_scope(대수선 8개
    기준 중 하나라도 해당하면 True), current_facility_group/desired_facility_group
    (1~9, FACILITY_GROUPS 참고), temporary_purpose, temporary_duration_years,
    temporary_is_concrete(철근콘크리트ᆞ철골조 여부).
    """
    act_type = facts.get("act_type")
    if act_type is None or _missing_fields(act_type, facts):
        return None

    thresholds = get_thresholds()

    if act_type in ("신축", "증축", "개축", "재축"):
        # 건축법 제14조: 증축ᆞ개축ᆞ재축은 상한 이내면 신고
        ext_limit = thresholds["증축개축재축_신고_상한_바닥면적_sqm"]
        if act_type in ("증축", "개축", "재축") and facts["extension_size_sqm"] <= ext_limit:
            return "건축신고"
        # 신축은 관리ᆞ농림ᆞ자연환경보전지역에서만 연면적ᆞ층수가 둘 다 기준 미만이면 신고
        new_build = thresholds["신축_신고"]
        if act_type == "신축" and facts["land_zone"] in new_build["대상_용도지역"]:
            if facts["size_sqm"] < new_build["연면적_미만_sqm"] and facts["floors"] < new_build["층수_미만"]:
                return "건축신고"
        return "건축허가"

    if act_type == "이전":
        return "건축허가"  # 제14조 신고 예외에 "이전"은 없음 - 원칙(제11조)대로 허가

    if act_type == "대수선":
        if not facts["renovation_scope"]:
            return "인허가불필요"  # 시행령 제3조의2 8개 기준 어디에도 안 걸림
        renov = thresholds["대수선_신고"]
        if facts["size_sqm"] < renov["연면적_미만_sqm"] and facts["floors"] < renov["층수_미만"]:
            return "건축신고"
        return "건축허가"

    if act_type == "용도변경":
        cur, dst = facts["current_facility_group"], facts["desired_facility_group"]
        if cur == dst:
            return "건축물대장기재변경"  # 제19조3항: 같은 시설군
        return "용도변경허가" if dst < cur else "용도변경신고"  # 번호 작을수록 상위군

    if act_type == "일반수선":
        return "인허가불필요"

    if act_type == "가설건축물":
        # 시행령 제15조: 존치기간 이내 + 비철콘조가 신고 요건. notice_purposes에
        # 속하는 목적은 그 자체로(기간ᆞ구조 무관) 신고 대상.
        temp = thresholds["가설건축물"]
        if facts["temporary_purpose"] in temp["신고_목적_예외"]:
            return "가설건축물신고"
        if facts["temporary_duration_years"] <= temp["신고_존치기간_이하_년"] and not facts["temporary_is_concrete"]:
            return "가설건축물신고"
        return "가설건축물허가"

    return None


# --- 구조화 출력 경계 (PermitResult) --------------------------------------
# 멀티 에이전트 설계서가 제안하는 Permit Agent의 구조화 출력 형태를 준비해두되,
# 아직 agent.py/api.py 어디에도 연결하지 않는다(Phase 2 - 내부 모델만 추가,
# 실제 연결은 Phase 5에서 Planner 도입과 함께 다시 논의). permit_type과
# procedures는 이미 classify_case()/PROCEDURE_TREE가 결정론적으로 갖고 있는
# 값이라 여기서도 규칙 기반으로 채운다 - LLM 판단으로 대체하지 않는다.
# required_documents/related_laws/explanation은 PROCEDURE_TREE에 없는(RAG 검색
# 결과가 필요한) 값이라 이 단계에서는 채우지 않고 LLM이 나중에 채울 자리로 남긴다.
class PermitResult(BaseModel):
    """Permit Agent의 구조화 출력. permit_type/procedures는 규칙 엔진 확정값,
    required_documents/related_laws/explanation은 LLM이 채울 자리(현재는 빈 값)."""

    permit_type: str = Field(description="classify_case()가 계산한 절차 결과(예: 건축허가)")
    procedures: list[str] = Field(description="PROCEDURE_TREE에서 결정론적으로 뽑은 절차 단계 라벨 목록")
    required_documents: list[str] = Field(default_factory=list, description="필수 서류 목록 - 아직 미채움")
    related_laws: list[str] = Field(default_factory=list, description="근거 법령 목록 - 아직 미채움")
    explanation: str = Field(default="", description="LLM이 생성할 설명 텍스트 - 아직 미채움")


def _walk_procedures(result_type: str, ownership: str | None) -> list[str]:
    """PROCEDURE_TREE의 root부터 stage 라벨을 순서대로 따라간다.

    분기(owner_check 등)를 만나면 ownership과 일치하는 선택지를 따라가고,
    모르면(None) 트리에 정의된 첫 선택지를 기본값으로 따라간다 - 두 선택지
    모두 결국 같은 다음 단계로 합류하는 구조라 결과 라벨 하나만 달라진다.
    """
    tree = PROCEDURE_TREE[result_type]
    nodes = tree["nodes"]
    labels: list[str] = []
    node_id: str | None = tree["root"]
    while node_id is not None:
        node = nodes[node_id]
        if node["type"] == "branch":
            choice = ownership if ownership in node["options"] else next(iter(node["options"]))
            node_id = node["options"][choice]
            continue
        labels.append(node["label"])
        node_id = node["next"]
    return labels


def build_permit_result(facts: dict) -> PermitResult | None:
    """case_facts로 PermitResult를 만든다. classify_case()가 아직 판정을 못 내리면
    (정보 부족) None을 반환 - 호출 쪽에서 이걸로 "아직 판정 전" 여부를 판단한다.
    """
    permit_type = classify_case(facts)
    if permit_type is None:
        return None
    return PermitResult(
        permit_type=permit_type,
        procedures=_walk_procedures(permit_type, facts.get("ownership")),
    )


def procedure_stage_message(result_type: str, node_id: str) -> str:
    """(result_type, node_id)에 해당하는 안내 문구를 만든다.

    LLM이 직접 부르는 도구가 아니라, agent.py의 classify_case 그래프 노드가
    분류 결과를 답변에 반영할 때 참고용으로 쓰는 헬퍼 함수 - 허가/신고 판정
    자체는 이제 규칙 기반이라 LLM이 이 판단을 직접 할 필요가 없어졌다.
    """
    tree = PROCEDURE_TREE.get(result_type)
    if tree is None or node_id not in tree["nodes"]:
        return f"'{node_id}'는 {result_type} 트리에 없는 노드입니다."

    node = tree["nodes"][node_id]
    if node["type"] == "branch":
        options = ", ".join(node["options"].keys())
        return f'현재 절차 결과: {result_type}. 다음을 사용자에게 물어보세요: "{node["question"]}" (선택지: {options})'
    return f"현재 절차 결과: {result_type} - {node['label']}."


def generate_mermaid(tree: dict = PROCEDURE_TREE) -> str:
    """PROCEDURE_TREE에서 mermaid flowchart 텍스트를 생성.

    트리가 유일한 원본이 되도록, 그림은 항상 이 함수로 새로 뽑아서 쓴다
    (손으로 그림과 코드를 따로 맞추지 않음).
    """
    lines = ["flowchart TD", '    Start(["대화 시작"]) --> CaseType{"케이스 유형?"}']

    for case_type, case in tree.items():
        prefix = case_type.replace(" ", "_")
        lines.append(f'    CaseType -->|{case_type}| {prefix}_{case["root"]}')

        for node_id, node in case["nodes"].items():
            full_id = f"{prefix}_{node_id}"
            if node["type"] == "branch":
                lines.append(f'    {full_id}{{"{node["question"]}"}}')
                for choice, target in node["options"].items():
                    lines.append(f'    {full_id} -->|{choice}| {prefix}_{target}')
            else:
                shape = f'(["{node["label"]}"])' if node["next"] is None else f'["{node["label"]}"]'
                lines.append(f'    {full_id}{shape}')
                if node["next"] is not None:
                    lines.append(f'    {full_id} --> {prefix}_{node["next"]}')

    return "\n".join(lines)


@tool
def record_case_facts(
    act_type: Literal[
        "신축", "증축", "개축", "재축", "이전", "대수선", "용도변경", "일반수선", "가설건축물"
    ] | None = None,
    size_sqm: float | None = None,
    floors: int | None = None,
    extension_size_sqm: float | None = None,
    land_zone: Literal["관리지역", "농림지역", "자연환경보전지역", "기타"] | None = None,
    renovation_scope: bool | None = None,
    current_facility_group: int | None = None,
    desired_facility_group: int | None = None,
    temporary_purpose: str | None = None,
    temporary_duration_years: float | None = None,
    temporary_is_concrete: bool | None = None,
    ownership: Literal["소유자", "임차인"] | None = None,
) -> str:
    """대화에서 파악된 사용자 상황 정보를 알게 되는 대로 기록하세요.

    - 이번 턴에 새로 알게 된 필드만 채우고 나머지는 생략(None)하세요. 여러 턴에
      걸쳐 나눠서 호출해도 이전에 기록한 값 위에 누적됩니다.
    - 허가/신고/기재변경 같은 판정은 여기서 계산 안 합니다 - 정보가 충분히
      모이면 시스템이 자동으로(규칙 기반) 판정하니, 사용자에게 "허가 대상인가요
      신고 대상인가요?" 같은 질문은 하지 마세요. 사용자가 실제로 답할 수 있는
      사실(면적/층수/용도/소유여부 등)만 물어보고 기록하세요.
    - search_regulations/search_by_term과 같은 턴에 함께 호출해도 됩니다.

    Args:
        act_type: 행위 유형
        size_sqm: 연면적(㎡) - 신축/대수선 판정용
        floors: 층수 - 신축/대수선 판정용
        extension_size_sqm: 증축ᆞ개축ᆞ재축 대상 부분의 바닥면적(㎡)
        land_zone: 용도지역 - 신축 판정용(관리ᆞ농림ᆞ자연환경보전지역인지 여부가 중요)
        renovation_scope: 대수선 정의(내력벽ᆞ기둥ᆞ보 등 8개 기준) 중 하나라도
            해당하면 True, 단순 마감재 교체 등은 False
        current_facility_group/desired_facility_group: 용도변경의 기존/희망 시설군
            번호(1~9, 자동차관련=1 ~ 그밖의시설군=9)
        temporary_purpose/temporary_duration_years/temporary_is_concrete: 가설건축물 관련
        ownership: 소유자 또는 임차인
    """
    logger.info("[도구] record_case_facts(%r)", {
        k: v for k, v in locals().items() if v is not None
    })
    return "기록됨."


record_case_facts.description += "\n\n시설군 매핑표(용도변경 시 current_facility_group/desired_facility_group에 이 번호를 쓰세요):\n" + "\n".join(
    f"{num}. {name}: {', '.join(items)}" for num, (name, items) in FACILITY_GROUPS.items()
)


@tool
def record_permit_synthesis(
    required_documents: list[str] | None = None,
    related_agencies: list[str] | None = None,
) -> str:
    """판정(permit_type)이 이미 확정된 뒤, 검색된 법령을 근거로 필요한 제출서류ᆞ
    협의기관을 파악하면 기록하세요.

    - record_case_facts와 마찬가지로 알게 되는 대로 부분적으로 호출해도
      이전에 기록한 값 위에 누적됩니다.
    - 아직 검색을 안 해서 모르면, 먼저 search_regulations/search_by_term으로
      근거를 확보한 뒤에 호출하세요 - 추측해서 채우지 마세요.
    """
    logger.info("[도구] record_permit_synthesis(%r)", {
        k: v for k, v in locals().items() if v is not None
    })
    return "기록됨."
