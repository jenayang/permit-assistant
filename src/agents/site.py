"""Site Agent - 주소 기반 부지 정보 자동 조회 + 건폐율ㆍ용적률 법정 상한 조회.

LLM 판단이 아니라 API 조회/정적 법령 표 중심으로 구현한다(설계서 3번 섹션 참고).
용도지역은 API 자동 조회(VWorld), 건폐율ㆍ용적률은 외부 API 없이 정적 표로
조회한다(Phase 3 - 도로폭은 이번 범위에서 제외, 별도 논의 예정). 건폐율ㆍ용적률
자체가 법령상 값이라 API 승인 여부와 무관하게 지금 바로 정확히 다룰 수 있고,
용도지역과 달리 "지구단위계획 등으로 완화될 수 있다"는 점만 안내하면 되기
때문 - classify_case()와 같은 이유로 LLM 판단에 맡기지 않는다.
"""
from __future__ import annotations

import logging
from typing import Literal

import requests
from langchain_core.tools import tool

from src import config

logger = logging.getLogger(__name__)


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


# --- 건폐율ㆍ용적률 법정 상한 (서울특별시 도시계획 조례 제44조ㆍ제48조 원문 대조 완료,
# 2026-07-22 - data/ordinances의 "서울특별시 도시계획 조례" PDF에서 직접 확인) ---
# 위 lookup_land_zone이 다루는 land_zone(관리ᆞ농림ᆞ자연환경보전지역ᆞ기타)과는 다른
# 분류축이다 - 이건 국토계획법상 "도시지역"을 세분한 용도지역(주거ᆞ상업ᆞ공업ᆞ녹지)
# 기준이라 별도 필드로 다룬다. 지구단위계획ㆍ완화조항(조례 제46ᆞ47ᆞ49~51조 등)은
# 반영하지 않은 기본 상한값이므로, 실제 확정치는 다를 수 있음을 항상 함께 안내한다.
BuildingZone = Literal[
    "제1종전용주거지역", "제2종전용주거지역",
    "제1종일반주거지역", "제2종일반주거지역", "제3종일반주거지역",
    "준주거지역",
    "중심상업지역", "일반상업지역", "근린상업지역", "유통상업지역",
    "전용공업지역", "일반공업지역", "준공업지역",
    "보전녹지지역", "생산녹지지역", "자연녹지지역",
]

BUILDING_RATIO_LIMITS: dict[str, dict[str, int]] = {
    "제1종전용주거지역": {"건폐율": 50, "용적률": 100},
    "제2종전용주거지역": {"건폐율": 40, "용적률": 120},
    "제1종일반주거지역": {"건폐율": 60, "용적률": 150},
    "제2종일반주거지역": {"건폐율": 60, "용적률": 200},
    "제3종일반주거지역": {"건폐율": 50, "용적률": 250},
    "준주거지역": {"건폐율": 60, "용적률": 400},
    "중심상업지역": {"건폐율": 60, "용적률": 1000, "용적률_서울도심": 800},
    "일반상업지역": {"건폐율": 60, "용적률": 800, "용적률_서울도심": 600},
    "근린상업지역": {"건폐율": 60, "용적률": 600, "용적률_서울도심": 500},
    "유통상업지역": {"건폐율": 60, "용적률": 600, "용적률_서울도심": 500},
    "전용공업지역": {"건폐율": 60, "용적률": 200},
    "일반공업지역": {"건폐율": 60, "용적률": 200},
    "준공업지역": {"건폐율": 60, "용적률": 400},
    "보전녹지지역": {"건폐율": 20, "용적률": 50},
    "생산녹지지역": {"건폐율": 20, "용적률": 50},
    "자연녹지지역": {"건폐율": 20, "용적률": 50},
}


@tool
def lookup_building_ratio_limits(zone: BuildingZone) -> str:
    """세부 용도지역명으로 건폐율ㆍ용적률 법정 상한(서울특별시 기준)을 조회합니다.

    - 사용자가 건폐율ㆍ용적률(예: "몇 층까지 지을 수 있나요", "대지면적 대비
      얼마나 크게 지을 수 있나요")을 물으면, 스스로 계산하거나 추측하지 말고
      이 도구를 호출하세요 - 법령상 정해진 값이라 classify_case와 같은 이유로
      LLM이 직접 계산하면 안 됩니다.
    - zone은 "제2종일반주거지역"처럼 세부 용도지역명이어야 합니다. 사용자가
      세부 용도지역을 모르면(대부분 모름) 직접 물어보세요 - lookup_land_zone은
      관리ᆞ농림ᆞ자연환경보전지역 여부만 구분하므로 이 조회에는 쓸 수 없습니다.
    - 결과에는 지구단위계획 등으로 완화될 수 있다는 안내가 포함되니 그대로
      전달하고, 최종 확정치는 구청 확인이 필요하다고 덧붙이세요.
    """
    logger.info("[도구] lookup_building_ratio_limits(zone=%r)", zone)
    limits = BUILDING_RATIO_LIMITS.get(zone)
    if limits is None:
        return f"'{zone}'에 대한 건폐율ㆍ용적률 기준을 찾을 수 없습니다."

    msg = f"{zone}의 건폐율 상한은 {limits['건폐율']}%, 용적률 상한은 {limits['용적률']}%입니다."
    if "용적률_서울도심" in limits:
        msg += f" (단, 서울도심 지역은 용적률 {limits['용적률_서울도심']}%)"
    msg += (
        " [출처: 서울특별시 도시계획 조례 제44조ㆍ제48조]"
        " 지구단위계획ㆍ완화 조항에 따라 실제 적용치는 달라질 수 있으니 최종 확정은 관할 구청에 확인하세요."
    )
    return msg
