"""역할별 에이전트 모듈 모음.

- regulation: 법령/조례/절차 문서 검색(RAG) - 판단하지 않고 검색만 담당.
- site: 주소 기반 부지 정보 자동 조회(용도지역 등, API 중심) + 건폐율ㆍ용적률 법정 상한 조회(정적 표).
- permit: 허가/신고/기재변경 판정(규칙 기반) + 절차 로드맵.

아직 별도 서브그래프/오케스트레이터는 아니고, agent.py의 단일 ReAct 루프가
이 모듈들을 도구로 바인딩해서 쓰는 구조. 향후 Planner 도입 시 이 경계를
그대로 서브에이전트 경계로 승격할 수 있도록 미리 나눠둔 것.
"""
from __future__ import annotations

from src.agents.permit import (
    FACILITY_GROUPS,
    PROCEDURE_TREE,
    classify_case,
    generate_mermaid,
    procedure_stage_message,
    record_case_facts,
)
from src.agents.regulation import search_by_term, search_regulations
from src.agents.site import lookup_building_ratio_limits, lookup_land_zone

TOOLS = [
    search_regulations,
    search_by_term,
    record_case_facts,
    lookup_land_zone,
    lookup_building_ratio_limits,
]
