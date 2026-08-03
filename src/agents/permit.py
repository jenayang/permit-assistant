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
#
# "대수선"은 size_sqm/floors를 여기 넣지 않는다 - 원래는 넣었었는데
# (2026-07-28 발견) renovation_scope=False(시행령 8개 기준 어디에도
# 해당 안 됨)면 애초에 대수선 자체가 아니라서 size_sqm/floors 값이 결과에
# 전혀 영향을 안 주는데도, 이 필드가 항상 필수라서 사용자가 그 값을 알려줄
# 때까지 챗봇이 계속 되물었다(docs/test_scenarios.md Part A 12번 케이스로
# 발견 - 문서엔 "size/floors 무관"이라고 이미 적혀 있었는데 코드가 그 스펙을
# 못 지키고 있었음). size_sqm/floors가 실제로 필요한 건 renovation_scope가
# True일 때뿐이라, 그 조건부 요구는 classify_case의 "대수선" 분기 안으로
# 옮겼다.
REQUIRED_FIELDS: dict[str, list[str]] = {
    "신축": ["size_sqm", "floors", "land_zone"],
    "증축": ["extension_size_sqm"],
    "개축": ["extension_size_sqm"],
    "재축": ["extension_size_sqm"],
    "이전": [],
    "대수선": ["renovation_scope"],
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


def facility_group_from_use_name(use_name: str) -> int | None:
    """건축물대장 주용도코드명(예: "제1종근린생활시설")을 시설군 번호(1~9)로
    매핑. 못 찾으면 None.

    FACILITY_GROUPS의 세부용도 목록과 대조하되 공백을 무시한다 - 대장은
    "제1종근린생활시설"(붙여씀), 표는 "제1종 근린생활시설"(띄어씀)이라 그대로는
    안 맞는다. 항목의 괄호 앞부분(핵심어)이 주용도명에 포함되면 그 시설군으로
    본다(단방향 포함만 - "제2종근린생활시설"이 group5의 "제2종 근린생활시설 중
    다중생활시설"에 역포함돼 오분류되는 걸 막기 위함)."""
    if not use_name:
        return None
    norm = use_name.replace(" ", "")
    for num, (_group_name, items) in FACILITY_GROUPS.items():
        for item in items:
            core = item.split("(")[0].replace(" ", "")  # "업무시설(사무실 등)" → "업무시설"
            if core and core in norm:
                return num
    return None


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
        # size_sqm/floors는 renovation_scope가 True일 때만 실제로 필요하다 -
        # REQUIRED_FIELDS엔 넣지 않았으니(위 주석 참고) 여기서 직접 확인한다.
        if facts.get("size_sqm") is None or facts.get("floors") is None:
            return None
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


def requires_licensed_architect(facts: dict) -> bool | None:
    """이 케이스가 건축사 설계 의무 대상인지 판정. 정보 부족하면 None.

    건축법 제23조1항: 건축허가(제11조)ᆞ건축신고(제14조) 대상 건축물의 설계는
    원칙적으로 건축사만 할 수 있으나 다음은 예외 -
      1호. 바닥면적 합계 85㎡ 미만 증축ᆞ개축ᆞ재축
      2호. 연면적 200㎡ 미만이고 층수 3층 미만인 대수선
    용도변경은 제19조6항이 별도로 규정 - "허가 대상(상위군 이동)이면서 용도변경
    부분 바닥면적 500㎡ 이상"인 경우에만 제23조를 준용한다.

    반환값:
      True  = 건축사사무소 설계가 필요(대행 대상)
      False = 개인이 직접 설계ᆞ진행 가능(제23조 예외이거나 제23조 비대상 행위)
      None  = 판정에 필요한 사실 부족

    주의: 제23조1항3호("그 밖에 대통령령으로 정하는 건축물")와 제23조4항
    (표준설계도서)은 시행령ᆞ국토부 고시가 있어야 확정되는데 현재 인덱스에 없어
    반영하지 않았다 - 그래서 "예외에 더 걸려 실제로는 건축사 불필요"인 케이스를
    True로 볼 여지가 남는다(안전한 방향의 과대판정). 또 용도변경의 바닥면적은
    facts에 별도 필드가 없어 size_sqm을 대용으로 쓴다 - "용도변경 부분"이 아니라
    건물 연면적일 수 있어 경계(500㎡) 부근에서 부정확할 수 있다.
    """
    act_type = facts.get("act_type")
    if act_type is None:
        return None
    thresholds = get_thresholds()

    if act_type == "신축":
        return True  # 제23조 예외 목록에 신축 없음 - 규모 무관 건축사 필요

    if act_type in ("증축", "개축", "재축"):
        ext = facts.get("extension_size_sqm")
        if ext is None:
            return None
        # 제23조1항1호: 85㎡ 미만이면 예외(건축사 불필요), 이상이면 필요
        return ext >= thresholds["증축개축재축_신고_상한_바닥면적_sqm"]

    if act_type == "이전":
        return True  # 건축허가 대상, 제23조 예외 없음

    if act_type == "대수선":
        if not facts.get("renovation_scope"):
            return False  # 대수선 아님(인허가불필요) - 제23조 비대상
        size, floors = facts.get("size_sqm"), facts.get("floors")
        if size is None or floors is None:
            return None
        renov = thresholds["대수선_신고"]
        # 제23조1항2호: 200㎡ 미만 & 3층 미만이면 예외(건축사 불필요)
        return not (size < renov["연면적_미만_sqm"] and floors < renov["층수_미만"])

    if act_type == "용도변경":
        cur, dst = facts.get("current_facility_group"), facts.get("desired_facility_group")
        if cur is None or dst is None:
            return None
        if dst >= cur:
            return False  # 신고 대상(하위군)ᆞ기재변경(동일군) - 제23조 준용 안 됨
        # 허가 대상(상위군): 제19조6항 - 바닥면적 500㎡ 이상만 제23조 준용
        area = facts.get("size_sqm")
        if area is None:
            return None
        return area >= thresholds["용도변경_건축사설계준용_상한_바닥면적_sqm"]

    if act_type in ("일반수선", "가설건축물"):
        # 일반수선은 허가ᆞ신고 대상 행위가 아니고, 가설건축물은 제20조 별도라
        # 둘 다 제23조(제11조ᆞ제14조 대상) 설계 의무 대상이 아니다.
        return False

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
    required_documents/pre_diagnosis_items/related_laws/explanation은 LLM이 채울 자리."""

    permit_type: str = Field(description="classify_case()가 계산한 절차 결과(예: 건축허가)")
    procedures: list[str] = Field(description="PROCEDURE_TREE에서 결정론적으로 뽑은 절차 단계 라벨 목록")
    required_documents: list[str] = Field(default_factory=list, description="필수 서류 목록 - 아직 미채움")
    pre_diagnosis_items: list[str] = Field(
        default_factory=list, description="사전 진단 항목(주차ᆞ정화조ᆞ소방 등) 목록 - 아직 미채움"
    )
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


def _facility_group_reason(result_type: str, facts: dict) -> str:
    """용도변경 판정 사유를 이미 확정된 result_type을 근거로 문장으로 서술한다.

    build_permit_result(facts)와 동일하게 facts 전체를 받는다 - 필요한 필드만
    콕 집어 파라미터로 받으면 다른 act_type(신축ᆞ증축 등)의 판정 사유를 추가할
    때마다 procedure_stage_message의 시그니처가 계속 늘어나야 해서(size_sqm,
    floors, land_zone, ...) 대신 facts를 통째로 넘기고 내부에서 result_type에
    맞는 키만 꺼내 쓰게 한다.

    classify_case()가 "dst < cur → 허가"를 판정할 때 이미 방향을 확정했으므로,
    여기서는 그 비교를 다시 하지 않고 result_type(있는 그대로의 결과)에 맞는
    설명만 붙인다 - 같은 비교를 두 곳에서 따로 하면 나중에 한쪽만 고쳐서
    서로 어긋날 수 있음(단일 진실 공급원 원칙). LLM에게 "번호가 작을수록
    상위군"이라는 규칙만 던져주고 스스로 방향을 재추론하게 하면 실제로
    반복해서 뒤집어 말하는 문제가 있었다(8→7을 "하위군 이동(신고)"으로
    착각 - 상위군 이동(허가)이 맞음, 2026-07-24 실측 확인) - 그래서 방향
    판단 자체를 LLM에게 맡기지 않고 이미 확정된 결과를 그대로 서술한다.
    """
    current_facility_group = facts.get("current_facility_group")
    desired_facility_group = facts.get("desired_facility_group")
    if current_facility_group is None or desired_facility_group is None:
        return ""
    if current_facility_group not in FACILITY_GROUPS or desired_facility_group not in FACILITY_GROUPS:
        return ""
    cur_label = f"{FACILITY_GROUPS[current_facility_group][0]}({current_facility_group})"
    dst_label = f"{FACILITY_GROUPS[desired_facility_group][0]}({desired_facility_group})"
    if result_type == "건축물대장기재변경":
        return f" (사유: {cur_label} 내에서의 변경)"
    if result_type == "용도변경허가":
        return f" (사유: {cur_label}→{dst_label}는 번호가 작아지는 상위군 이동 - 번호가 작을수록 상위군)"
    if result_type == "용도변경신고":
        return f" (사유: {cur_label}→{dst_label}는 번호가 커지는 하위군 이동 - 번호가 작을수록 상위군)"
    return ""


def _act_type_reason(result_type: str, facts: dict) -> str:
    """용도변경 외 나머지 act_type(신축ᆞ증축ᆞ개축ᆞ재축ᆞ이전ᆞ대수선ᆞ일반수선ᆞ
    가설건축물)의 판정 사유를 문장으로 서술한다. _facility_group_reason과 같은
    원칙 - classify_case()가 이미 확정한 result_type을 다시 판정하지 않고
    사유만 서술한다. 숫자는 get_thresholds()를 그대로 참조해서 법정 임계값이
    바뀌어도 이 함수를 따로 안 고쳐도 되고, 비교 부등호(<, <=)는 classify_case()
    와 반드시 같은 방향으로 맞춰뒀다(재-비교이므로 어긋나면 사유와 판정이
    서로 다른 말을 하게 됨 - 단일 진실 공급원 원칙, _facility_group_reason 참고).

    2026-07-29: 지금까지 용도변경만 "왜?" 사유 문장이 있었고 나머지 act_type은
    LLM이 즉흥적으로 설명해야 했다(RAG 활용도 점검 중 사용자 피드백으로 발견) -
    판정 신뢰성 원칙을 전체 act_type으로 넓힌다.
    """
    act_type = facts.get("act_type")
    thresholds = get_thresholds()

    if act_type in ("증축", "개축", "재축"):
        limit = thresholds["증축개축재축_신고_상한_바닥면적_sqm"]
        size = facts.get("extension_size_sqm")
        if size is None:
            return ""
        if size <= limit:
            return f" (사유: {act_type} 대상 부분 바닥면적 {size}㎡가 {limit}㎡ 이내)"
        return f" (사유: {act_type} 대상 부분 바닥면적 {size}㎡가 {limit}㎡ 초과)"

    if act_type == "신축":
        new_build = thresholds["신축_신고"]
        zone = facts.get("land_zone")
        size, floors = facts.get("size_sqm"), facts.get("floors")
        if zone not in new_build["대상_용도지역"]:
            # "관리지역"ᆞ"농림지역"ᆞ"자연환경보전지역"은 받침 있음(은), "기타"는
            # 받침 없음(는) - Literal 값 4개 중 "기타"만 예외라 하드코딩 분기.
            particle = "는" if zone == "기타" else "은"
            return f" (사유: 용도지역 '{zone}'{particle} 관리ᆞ농림ᆞ자연환경보전지역이 아니라 신고 예외 대상이 아님)"
        if size is None or floors is None:
            return ""
        limit_size, limit_floors = new_build["연면적_미만_sqm"], new_build["층수_미만"]
        if size < limit_size and floors < limit_floors:
            return f" (사유: {zone}이고 연면적 {size}㎡ᆞ{floors}층이 각각 {limit_size}㎡ᆞ{limit_floors}층 미만 기준을 충족)"
        return f" (사유: {zone}이지만 연면적ᆞ층수 기준({limit_size}㎡ᆞ{limit_floors}층 미만) 중 하나 이상을 초과)"

    if act_type == "이전":
        return " (사유: 건축법 제14조 신고 예외 목록에 '이전'이 없어 원칙(제11조)대로 허가 대상)"

    if act_type == "대수선":
        if facts.get("renovation_scope") is False:
            return " (사유: 대수선 정의(시행령 제3조의2 8개 기준) 중 어디에도 해당하지 않음)"
        renov = thresholds["대수선_신고"]
        size, floors = facts.get("size_sqm"), facts.get("floors")
        if size is None or floors is None:
            return ""
        limit_size, limit_floors = renov["연면적_미만_sqm"], renov["층수_미만"]
        if size < limit_size and floors < limit_floors:
            return f" (사유: 대수선 정의에 해당하고 연면적 {size}㎡ᆞ{floors}층이 각각 {limit_size}㎡ᆞ{limit_floors}층 미만 기준을 충족)"
        return f" (사유: 대수선 정의에 해당하지만 연면적ᆞ층수 기준({limit_size}㎡ᆞ{limit_floors}층 미만) 중 하나 이상을 초과)"

    if act_type == "일반수선":
        return " (사유: 일반수선은 대수선ᆞ신고ᆞ허가 대상 행위에 해당하지 않음)"

    if act_type == "가설건축물":
        temp = thresholds["가설건축물"]
        purpose = facts.get("temporary_purpose")
        duration = facts.get("temporary_duration_years")
        is_concrete = facts.get("temporary_is_concrete")
        if purpose in temp["신고_목적_예외"]:
            return f" (사유: '{purpose}' 목적은 존치기간ᆞ구조와 무관하게 신고 대상(시행령 제15조 예외 목적))"
        if duration is None or is_concrete is None:
            return ""
        limit_years = temp["신고_존치기간_이하_년"]
        if duration <= limit_years and not is_concrete:
            return f" (사유: 존치기간 {duration}년이 {limit_years}년 이내이고 비철근콘크리트ᆞ철골조라 신고 요건 충족)"
        if duration > limit_years:
            return f" (사유: 존치기간 {duration}년이 {limit_years}년을 초과해 신고 요건 미충족)"
        return " (사유: 철근콘크리트ᆞ철골조라 신고 요건(비철콘조) 미충족)"

    return ""


# 설계도서(평면도 등)가 실제로 필요한 판정 결과 - 이 경우에만 건축사 의무ᆞ실무
# 도움 안내가 의미 있다(인허가불필요ᆞ기재변경 등엔 도면 부담이 없어 제외).
_RESULTS_NEEDING_DESIGN_DOCS = frozenset({
    "건축신고", "건축허가", "용도변경신고", "용도변경허가",
})


def _architect_note(result_type: str, facts: dict) -> str:
    """도구 응답에 실어보낼 건축사 설계 의무 안내(결정론적). LLM은 이 문장을
    새로 판단하지 않고 그대로 전달만 한다 - 판정은 코드, 설명은 LLM 원칙.
    설계도서가 필요없는 결과이거나 정보가 부족하면 빈 문자열."""
    if result_type not in _RESULTS_NEEDING_DESIGN_DOCS:
        return ""
    verdict = requires_licensed_architect(facts)
    if verdict is None:
        return ""
    if verdict:
        return " 설계도서는 건축법 제23조에 따라 건축사사무소에서 작성해야 합니다(건축사 설계 의무 대상)."
    return (
        " 건축사 설계 의무 대상은 아니어서 법적으로는 개인이 직접 진행할 수 있으나, "
        "평면도 등 설계도서 준비가 필요해 실무에서는 인테리어 업체나 행정사의 도움을 "
        "받는 경우가 많습니다."
    )


def procedure_stage_message(result_type: str, node_id: str, facts: dict | None = None) -> str:
    """(result_type, node_id)에 해당하는 안내 문구를 만든다.

    LLM이 직접 부르는 도구가 아니라, agent.py의 classify_case 그래프 노드가
    분류 결과를 답변에 반영할 때 참고용으로 쓰는 헬퍼 함수 - 허가/신고 판정
    자체는 이제 규칙 기반이라 LLM이 이 판단을 직접 할 필요가 없어졌다.
    판정 사유(예: 용도변경의 시설군 이동 방향, 신축/대수선의 규모 기준 충족
    여부 등)까지 이미 확정된 result_type 기준으로 미리 문장을 만들어 같이
    넘긴다(facts 전체를 받는 이유는 _facility_group_reason 참고). 용도변경은
    _facility_group_reason이, 나머지 act_type은 _act_type_reason이 담당한다.
    """
    tree = PROCEDURE_TREE.get(result_type)
    if tree is None or node_id not in tree["nodes"]:
        return f"'{node_id}'는 {result_type} 트리에 없는 노드입니다."

    reason = _facility_group_reason(result_type, facts or {}) or _act_type_reason(result_type, facts or {})
    architect = _architect_note(result_type, facts or {})
    node = tree["nodes"][node_id]
    if node["type"] == "branch":
        options = ", ".join(node["options"].keys())
        return (
            f'현재 절차 결과: {result_type}{reason}. 다음을 사용자에게 물어보세요: '
            f'"{node["question"]}" (선택지: {options}){architect}'
        )
    return f"현재 절차 결과: {result_type}{reason} - {node['label']}.{architect}"


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
    pre_diagnosis_items: list[str] | None = None,
) -> str:
    """판정(permit_type)이 이미 확정된 뒤, 검색된 법령을 근거로 필요한 제출서류ᆞ
    협의기관ᆞ사전 진단 항목을 파악하면 기록하세요.

    - record_case_facts와 마찬가지로 알게 되는 대로 부분적으로 호출해도
      이전에 기록한 값 위에 누적됩니다.
    - 아직 검색을 안 해서 모르면, 먼저 search_regulations/search_by_term으로
      근거를 확보한 뒤에 호출하세요 - 추측해서 채우지 마세요.
    - pre_diagnosis_items: [3단계: 사전 진단]에서 안내한 항목(예: "주차대수
      기준 확인", "정화조 용량 확인")을 답변과 동일한 문구로 기록하세요 -
      프론트엔드가 이 목록의 존재 여부로 3단계 완료를 판단합니다(빈 목록이면
      아직 3단계 진행 중으로 취급됨).
    """
    logger.info("[도구] record_permit_synthesis(%r)", {
        k: v for k, v in locals().items() if v is not None
    })
    return "기록됨."
