"""Site Agent - 주소 기반 부지 정보 자동 조회.

LLM 판단이 아니라 API 조회 중심으로 구현한다(설계서 3번 섹션 참고).
현재는 용도지역 자동 조회만 구현되어 있고, 건폐율ᆞ용적률ᆞ도로폭 등은
향후 확장 대상(Phase 3).
"""
from __future__ import annotations

import logging

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
