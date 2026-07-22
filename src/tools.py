"""에이전트가 사용할 도구 정의.

@tool 데코레이터로 정의된 함수는 LLM이 자동으로 호출 가능.
docstring이 LLM에게 도구 선택 힌트 - 명확하게 작성 필수.
"""
from __future__ import annotations

import logging
from typing import Literal

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

# 케이스 유형별 실제 인허가 절차 단계를 분기 포함 트리로 표현. data/procedures/엔
# 케이스타입별 공식 세부 절차 문서가 없어서(범용 허가~사용승인 흐름 PDF 1개뿐)
# 의도적으로 단순화한 근사치임 - UI 쪽에 "참고용" 안내 문구를 별도로 붙인다
# (프롬프트가 아니라 정적 캡션으로).
#
# mermaid 다이어그램은 이 dict에서 generate_mermaid()로 생성하므로, 트리를
# 고치면 그림도 같이 최신 상태로 유지된다 - 손으로 안 맞춰도 됨.
#
# 노드 형태:
#   분기: {"type": "branch", "question": "...", "options": {"선택지": "다음노드id"}}
#   단계: {"type": "stage", "label": "...", "next": "다음노드id" | None(=완료/종료)}
PROCEDURE_TREE: dict[str, dict] = {
    "신축": {
        "root": "permit_type",
        "nodes": {
            "permit_type": {
                "type": "branch", "question": "허가 대상인가요, 신고 대상인가요?",
                "options": {"신고 대상": "construction_notice", "허가 대상": "owner_check"},
            },
            "owner_check": {
                "type": "branch", "question": "소유자이신가요, 임차인이신가요?",
                "options": {"소유자": "permit_apply_owner", "임차인": "permit_apply_tenant"},
            },
            "permit_apply_owner": {"type": "stage", "label": "허가 신청 (소유권 증빙)", "next": "construction_notice"},
            "permit_apply_tenant": {"type": "stage", "label": "허가 신청 (토지사용승낙서 등)", "next": "construction_notice"},
            "construction_notice": {"type": "stage", "label": "착공신고", "next": "construction"},
            "construction": {"type": "stage", "label": "착공/시공", "next": "approval"},
            "approval": {"type": "stage", "label": "사용승인", "next": "done"},
            "done": {"type": "stage", "label": "완료", "next": None},
        },
    },
    "용도변경": {
        "root": "same_group",
        "nodes": {
            "same_group": {
                "type": "branch", "question": "기존 용도와 같은 시설군인가요?",
                "options": {"같음": "registry_change", "다름": "permit_or_report"},
            },
            "registry_change": {"type": "stage", "label": "건축물대장 기재사항 변경 신청", "next": "done"},
            "permit_or_report": {
                "type": "branch", "question": "신고 대상인가요, 허가 대상인가요?",
                "options": {"신고 대상": "owner_check", "허가 대상": "owner_check"},
            },
            "owner_check": {
                "type": "branch", "question": "소유자이신가요, 임차인이신가요?",
                "options": {"소유자": "docs_owner", "임차인": "docs_tenant"},
            },
            "docs_owner": {"type": "stage", "label": "서류 제출 (소유권 증빙)", "next": "done"},
            "docs_tenant": {"type": "stage", "label": "서류 제출 (임대차계약서 + 소유자 동의서)", "next": "done"},
            "done": {"type": "stage", "label": "완료(용도변경 반영)", "next": None},
        },
    },
    "대수선": {
        "root": "scope_check",
        "nodes": {
            "scope_check": {
                "type": "branch", "question": "대수선 범위에 해당하나요?",
                "options": {"해당": "type_check", "미해당": "not_applicable"},
            },
            "not_applicable": {"type": "stage", "label": "대수선 아님 - 별도 절차 확인 필요", "next": None},
            "type_check": {
                "type": "branch", "question": "신고 대상인가요, 허가 대상인가요?",
                "options": {"신고 대상": "owner_check", "허가 대상": "owner_check"},
            },
            "owner_check": {
                "type": "branch", "question": "소유자이신가요, 임차인이신가요?",
                "options": {"소유자": "apply_owner", "임차인": "apply_tenant"},
            },
            "apply_owner": {"type": "stage", "label": "신고/허가 신청 (소유권 증빙)", "next": "construction"},
            "apply_tenant": {"type": "stage", "label": "신고/허가 신청 (소유자 동의서)", "next": "construction"},
            "construction": {"type": "stage", "label": "착공/시공", "next": "done"},
            "done": {"type": "stage", "label": "완료", "next": None},
        },
    },
}


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


def _describe_tree(tree: dict) -> str:
    lines = []
    for case_type, case in tree.items():
        lines.append(f'- {case_type} (시작 노드: "{case["root"]}"):')
        for node_id, node in case["nodes"].items():
            if node["type"] == "branch":
                opts = ", ".join(f'"{k}"→"{v}"' for k, v in node["options"].items())
                lines.append(f'    "{node_id}" [분기 질문: "{node["question"]}"] {opts}')
            else:
                nxt = f'"{node["next"]}"' if node["next"] else "(종료)"
                lines.append(f'    "{node_id}" [단계: "{node["label"]}"] → {nxt}')
    return "\n".join(lines)


@tool
def set_procedure_stage(
    case_type: Literal["신축", "용도변경", "대수선"],
    node_id: str,
) -> str:
    """사용자 상황에서 케이스 유형이 파악되면, 트리(아래 참고)를 따라가면서
    현재 도달한 노드를 호출해 UI 로드맵에 기록하세요.

    - 도달한 노드가 "분기"면, 답변 텍스트에 그 분기 질문을 사용자에게
      반드시 물어보세요(선택지도 함께 안내). 사용자가 답하면 그 선택지에
      해당하는 다음 노드로 넘어가서 다시 호출하세요.
    - 도달한 노드가 "단계"면 평소처럼 답변을 이어가세요.
    - 대화 이력에 직전과 같은 노드를 이미 보고했다면 다시 호출하지 마세요.
    - search_regulations/search_by_term과 같은 턴에 함께 호출해도 됩니다.
    - 이 호출 자체는 내부 기록용이니 "기록했습니다" 같은 말은 답변에 넣지 마세요.
    """
    logger.info("[도구] set_procedure_stage(case_type=%r, node_id=%r)", case_type, node_id)

    tree = PROCEDURE_TREE.get(case_type)
    if tree is None or node_id not in tree["nodes"]:
        valid_ids = list(tree["nodes"].keys()) if tree else []
        return f"'{node_id}'는 {case_type} 트리에 없는 노드입니다. 유효 노드: {', '.join(valid_ids)}"

    node = tree["nodes"][node_id]
    if node["type"] == "branch":
        options = ", ".join(node["options"].keys())
        return (
            f'기록됨(분기 노드). 답변에서 반드시 다음 질문을 사용자에게 물어보세요: '
            f'"{node["question"]}" (선택지: {options})'
        )
    return "기록됨. 이어서 사용자 질문에 대한 답변을 이어가세요."


set_procedure_stage.description += "\n\n트리 구조:\n" + _describe_tree(PROCEDURE_TREE)


TOOLS = [search_regulations, search_by_term, set_procedure_stage]
