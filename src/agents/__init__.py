"""역할별 에이전트 모듈 모음.

- regulation: 법령/조례/절차 문서 검색(RAG) - 판단하지 않고 검색만 담당.
- site: 주소 기반 부지 정보 자동 조회(용도지역ᆞ건축물대장 등, API 중심) + 건폐율ㆍ용적률 법정 상한 조회(정적 표).
- permit: 허가/신고/기재변경 판정(규칙 기반) + 절차 로드맵.
- food_safety: 식품위생법상 영업 종류 판정(규칙 기반) - 카페ᆞ음식점 등 식품접객업 창업 시 영업신고 종류.
- fire_safety: 소방시설 설치 대상 판정(규칙 기반) - 연면적 등 기준으로 필요한 소방시설 목록.

아직 별도 서브그래프/오케스트레이터는 아니고, agent.py의 단일 ReAct 루프가
이 모듈들을 도구로 바인딩해서 쓰는 구조. 향후 Planner 도입 시 이 경계를
그대로 서브에이전트 경계로 승격할 수 있도록 미리 나눠둔 것.
"""
from __future__ import annotations

from src.agents.fire_safety import classify_fire_safety, fire_safety_message, record_fire_facts
from src.agents.food_safety import classify_food_business, food_business_message, record_food_facts
from src.agents.permit import (
    FACILITY_GROUPS,
    PROCEDURE_TREE,
    PermitResult,
    build_permit_result,
    classify_case,
    generate_mermaid,
    procedure_stage_message,
    record_case_facts,
    record_permit_synthesis,
)
from src.agents.regulation import search_by_term, search_regulations
from src.agents.site import lookup_building_ledger, lookup_building_ratio_limits, lookup_land_zone

TOOLS = [
    search_regulations,
    search_by_term,
    record_case_facts,
    lookup_land_zone,
    lookup_building_ratio_limits,
    lookup_building_ledger,
    record_permit_synthesis,
    record_food_facts,
    record_fire_facts,
]
