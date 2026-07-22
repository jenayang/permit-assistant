"""에이전트가 사용할 도구 정의.

@tool 데코레이터로 정의된 함수는 LLM이 자동으로 호출 가능.
docstring이 LLM에게 도구 선택 힌트 - 명확하게 작성 필수.
"""
from __future__ import annotations

import logging
from typing import Literal

import requests
from langchain_core.tools import tool

from src import config
from src import parent_store
from src.retriever import keyword_search, search_with_score

logger = logging.getLogger(__name__)


# 카테고리별 한글 이름 매핑
CATEGORY_NAMES = {
    "laws": "법령",
    "ordinances": "서울시 조례",
    "procedures": "절차 안내",
}

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
# 신뢰할 수 없다(오늘 실제로 도구 호출을 빼먹는 문제를 겪음). 여기 쓰인 수치는 모두
# 건축법/시행령 원문 대조 완료(2026-07-22 세션).
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

    if act_type in ("신축", "증축", "개축", "재축"):
        # 건축법 제14조: 증축ᆞ개축ᆞ재축은 85㎡ 이내면 신고
        if act_type in ("증축", "개축", "재축") and facts["extension_size_sqm"] <= 85:
            return "건축신고"
        # 신축은 관리ᆞ농림ᆞ자연환경보전지역에서만 200㎡ 미만+3층 미만이면 신고
        if act_type == "신축" and facts["land_zone"] in ("관리지역", "농림지역", "자연환경보전지역"):
            if facts["size_sqm"] < 200 and facts["floors"] < 3:
                return "건축신고"
        return "건축허가"

    if act_type == "이전":
        return "건축허가"  # 제14조 신고 예외에 "이전"은 없음 - 원칙(제11조)대로 허가

    if act_type == "대수선":
        if not facts["renovation_scope"]:
            return "인허가불필요"  # 시행령 제3조의2 8개 기준 어디에도 안 걸림
        if facts["size_sqm"] < 200 and facts["floors"] < 3:
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
        # 시행령 제15조: 존치 3년 이내 + 비철콘조가 신고 요건. 재해복구ᆞ전시박람회ᆞ
        # 공사용ᆞ비닐하우스ᆞ컨테이너 등은 그 자체로 신고 대상.
        notice_purposes = {"재해복구", "전시박람회", "공사용", "견본주택", "비닐하우스", "컨테이너"}
        if facts["temporary_purpose"] in notice_purposes:
            return "가설건축물신고"
        if facts["temporary_duration_years"] <= 3 and not facts["temporary_is_concrete"]:
            return "가설건축물신고"
        return "가설건축물허가"

    return None


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
def search_regulations(query:str) -> str:
    """건축 인허가 관련 법령/조례/절차 문서를 검색합니다.
    
    다음과 같은 경우에 사용하세요:
    - 건축법, 시행령, 시행규칙 관련 조항
    - 서울시 건축조례, 도시계획조례
    - 건축신고, 건축허가, 용도변경 절차
    - 근린생활시설(카페, 사무실 등) 관련 규정
    - 주차, 정화조, 소방 등 부대시설 요구사항
    - 필요 서류 및 처리 기간
    
    Args:
        query: 검색할 키워드나 질문 (한국어)
    
    Returns:
        관련 법령/조례 조항의 요약 텍스트.
    """
    logger.info("[도구] search_regulations(query=%r)", query)

    results = search_with_score(query, k=config.TOP_K)

    if not results:
        return "관련 규정를 찾을 수 없습니다."

    # 부모-자식 청킹: 검색은 작은 자식 청크로 하되, LLM에는 그 조항 전체를 보여줌.
    # 같은 조항의 자식 청크 여러 개가 매칭되면 한 번만 포함(중복 제거).
    seen_articles: set[tuple[str, str]] = set()
    parts = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        article_id = doc.metadata.get("article_id")

        if article_id is not None:
            key = (source, article_id)
            if key in seen_articles:
                continue
            seen_articles.add(key)

        category = doc.metadata.get("category", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")  # 두번째는 디폴트값
        similarity = 1 - score
        category_kr = CATEGORY_NAMES.get(category, category)

        content = doc.page_content
        if article_id is not None:
            parent_content = parent_store.get_parent(source, article_id)
            if parent_content is not None:
                content = parent_content

        parts.append(
            f"[문서 {len(parts) + 1} | {category_kr}, 청크: {chunk_idx}, 유사도: {similarity:.3f}]\n"
            f"{content}"
        )

    return "\n\n".join(parts)


@tool
def search_by_term(keyword: str) -> str:
    """법률 용어의 정의/뜻을 물어보는 질문에 사용하는 키워드(정확 매칭) 검색 도구.

    예: "건폐율이 뭐야", "용적률의 정의는?", "이격거리란 무엇인가요"
    -> 이런 질문에서는 핵심 법률 용어(예: "건폐율")만 추출해서 이 도구에 전달하세요.

    search_regulations(유사도 검색)는 조문이 딱딱한 문어체라 "~란 무엇인가요"
    같은 자연어 질문과 의미 유사도가 낮게 나와 정의 조항을 놓칠 수 있습니다.
    이 도구는 용어를 포함한 조항을 부분 문자열로 정확히 찾아냅니다.

    Args:
        keyword: 찾고자 하는 핵심 법률 용어 (예: "건폐율", "용적률", "대지면적")

    Returns:
        해당 용어가 포함된 조항들의 텍스트 (정의 조항 우선 정렬).
    """
    logger.info("[도구] search_by_term(keyword=%r)", keyword)

    docs = keyword_search(keyword)

    if not docs:
        return f"'{keyword}'를 포함한 조항을 찾을 수 없습니다."

    parts = []
    for i, doc in enumerate(docs, 1):
        category = doc.metadata.get("category", "unknown")
        article_id = doc.metadata.get("article_id", "?")
        category_kr = CATEGORY_NAMES.get(category, category)

        parts.append(
            f"[문서 {i} | {category_kr}, 조항: 제{article_id}조]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(parts)


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


# --- 용도지역 자동 조회 (브이월드 지오코더 + 2D데이터 API) ------------------
# 일반 사용자는 자기 땅의 용도지역(관리ᆞ농림ᆞ자연환경보전지역 여부)을 모르는
# 경우가 많아서, 매번 사용자에게 직접 물어보게 하는 대신 주소 기반으로 자동
# 조회를 시도한다. 2026-07-22 기준: 지오코더는 개발키로 실제 호출해서 응답
# 구조를 확인했지만, 2D데이터 API는 운영키 승인 전이라 INCORRECT_KEY로 막혀있어
# 실제 응답을 못 봤다 - 아래 _vworld_query_land_zone의 레이어ID/속성 필드명은
# 학습 시점 지식 기반 추정치이므로 운영키 승인 후 실제 응답으로 재검증 필요.
_VWORLD_BASE = "https://api.vworld.kr/req"


def _vworld_geocode(address: str) -> tuple[float, float] | None:
    """주소 → (경도 x, 위도 y). 도로명/지번 둘 다 시도(사용자가 어느 형식으로
    말할지 모름). 실패하면 None - 호출부에서 사용자에게 재질문하도록 유도.
    """
    for addr_type in ("road", "parcel"):
        try:
            resp = requests.get(
                f"{_VWORLD_BASE}/address",
                params={
                    "service": "address",
                    "request": "getcoord",
                    "version": "2.0",
                    "crs": "epsg:4326",
                    "address": address,
                    "format": "json",
                    "type": addr_type,
                    "key": config.VWORLD_API_KEY,
                },
                timeout=5,
            )
            response = resp.json().get("response", {})
            if response.get("status") == "OK":
                point = response["result"]["point"]
                return float(point["x"]), float(point["y"])
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning("[lookup_land_zone] 지오코딩 실패(type=%s): %s", addr_type, exc)
    return None


def _map_land_zone_category(raw_name: str) -> str:
    """국토계획법상 관리ᆞ농림ᆞ자연환경보전지역은 세부 명칭에 그 단어가 그대로
    들어가므로(계획관리지역ᆞ생산관리지역ᆞ보전관리지역 등), 정확한 응답 필드
    구조를 몰라도 부분 문자열 매칭이면 웬만해선 안전하다.
    """
    if "관리지역" in raw_name:
        return "관리지역"
    if "농림지역" in raw_name:
        return "농림지역"
    if "자연환경보전지역" in raw_name:
        return "자연환경보전지역"
    return "기타"


def _vworld_query_land_zone(x: float, y: float) -> str | None:
    """TODO(운영키 승인 후 검증): data=LT_C_UQ111(용도지역지구도_용도지역 추정)와
    응답 속성 키(현재는 첫 속성값을 그냥 사용) 둘 다 미검증. 운영키로 실제 호출해
    원본 응답(로그의 "2D데이터 API 원본 응답")을 보고 정확한 속성 키로 고칠 것.
    """
    try:
        resp = requests.get(
            f"{_VWORLD_BASE}/data",
            params={
                "service": "data",
                "request": "GetFeature",
                "data": "LT_C_UQ111",
                "key": config.VWORLD_API_KEY,
                "geomFilter": f"POINT({x} {y})",
                "geometry": "false",
                "attribute": "true",
                "crs": "EPSG:4326",
                "format": "json",
            },
            timeout=5,
        )
        data = resp.json()
        logger.info("[lookup_land_zone] 2D데이터 API 원본 응답(검증용): %r", data)
        response = data.get("response", {})
        if response.get("status") != "OK":
            logger.warning("[lookup_land_zone] 2D데이터 API 실패: %s", response.get("error"))
            return None
        features = response["result"]["featureCollection"]["features"]
        if not features:
            return None
        raw_name = str(next(iter(features[0]["properties"].values())))
        return _map_land_zone_category(raw_name)
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        logger.warning("[lookup_land_zone] 2D데이터 조회 실패: %s", exc)
        return None


@tool
def lookup_land_zone(address: str) -> str:
    """주소로 용도지역(관리ᆞ농림ᆞ자연환경보전지역 여부)을 자동 조회합니다.

    - 신축/대수선 판정에 용도지역이 필요한데 사용자가 구/동 이상 수준의 주소를
      언급했다면, "용도지역이 뭔가요?"라고 사용자에게 직접 묻지 말고 먼저 이
      도구를 호출하세요(일반인은 자기 땅 용도지역을 모르는 경우가 많음).
    - 조회에 성공하면 결과를 그 턴에 record_case_facts(land_zone=...)로 바로
      기록하세요.
    - "확인 불가" 응답이 오면(자동 조회 실패) 그때만 사용자에게 직접 물어보세요.
    """
    if not config.VWORLD_API_KEY:
        return "용도지역 자동 조회가 설정되어 있지 않습니다(API 키 없음). 사용자에게 용도지역을 직접 물어보세요."

    logger.info("[도구] lookup_land_zone(address=%r)", address)
    coord = _vworld_geocode(address)
    if coord is None:
        return f"'{address}' 주소를 좌표로 변환하지 못했습니다. 사용자에게 더 정확한 주소를 요청하거나, 용도지역을 직접 물어보세요."

    zone = _vworld_query_land_zone(*coord)
    if zone is None:
        return "용도지역 자동 조회에 실패했습니다(서비스 오류 또는 아직 운영키 미승인). 사용자에게 용도지역을 직접 물어보세요."

    return f"'{address}'의 용도지역은 {zone}입니다. 이번 턴에 record_case_facts(land_zone='{zone}')로 기록하세요."


TOOLS = [search_regulations, search_by_term, record_case_facts, lookup_land_zone]
