"""Site Agent - 주소 기반 부지 정보 자동 조회 + 건폐율ㆍ용적률 법정 상한 조회.

LLM 판단이 아니라 API 조회/정적 법령 표 중심으로 구현한다(설계서 3번 섹션 참고).
용도지역은 API 자동 조회(VWorld), 건폐율ㆍ용적률은 외부 API 없이 정적 표로
조회한다(Phase 3 - 도로폭은 이번 범위에서 제외, 별도 논의 예정). 건폐율ㆍ용적률
자체가 법령상 값이라 API 승인 여부와 무관하게 지금 바로 정확히 다룰 수 있고,
용도지역과 달리 "지구단위계획 등으로 완화될 수 있다"는 점만 안내하면 되기
때문 - classify_case()와 같은 이유로 LLM 판단에 맡기지 않는다.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from langchain_core.tools import tool
from PublicDataReader import BuildingLedger

from src import config
from src.agents.legal_data import get_thresholds

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
# 실제 수치는 permit_thresholds.yaml(건폐율_용적률_상한)에 있음 - permit.py의
# classify_case()가 쓰는 법정 임계값과 같은 파일을 공유해서, 법령 개정 시 이
# 파이썬 파일을 안 건드리고 YAML만 고치면 되게 한다.
BuildingZone = Literal[
    "제1종전용주거지역", "제2종전용주거지역",
    "제1종일반주거지역", "제2종일반주거지역", "제3종일반주거지역",
    "준주거지역",
    "중심상업지역", "일반상업지역", "근린상업지역", "유통상업지역",
    "전용공업지역", "일반공업지역", "준공업지역",
    "보전녹지지역", "생산녹지지역", "자연녹지지역",
]


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
    limits = get_thresholds()["건폐율_용적률_상한"].get(zone)
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


# --- 건축물대장 표제부 자동 조회 (건축HUB 건축물대장정보 서비스) --------------
# 용도변경ᆞ대수선ᆞ증축처럼 "기존 건물"이 있는 케이스에서, 그 건물의 현재 등록
# 정보(용도ᆞ연면적ᆞ층수ᆞ건폐율ᆞ용적률ᆞ사용승인일)를 사용자에게 직접 묻는 대신
# 자동 조회한다. 신축(빈 땅)에는 대장이 없으므로 이 도구는 그 경우 쓸모없다.
#
# 건축HUB는 주소가 아니라 (시군구코드, 법정동코드, 번지)로 조회하므로, 먼저 전국
# 법정동코드 표(공개 데이터, WooilJeong/code 저장소)를 내려받아 캐싱해두고 주소
# 텍스트와 이름을 매칭해서 코드를 찾는다. PublicDataReader(오픈소스 라이브러리)로
# API 엔드포인트 URL과 영문→한글 컬럼 매핑을 가져다 쓴다(2026-07-23 실제 설치해서
# meta_dict/translate_columns 결과 직접 확인 완료 - 필드명 추측 아님).
_BDONG_JSON_URL = "https://raw.githubusercontent.com/WooilJeong/code/main/code/code_dong/code_bdong.json"
_BDONG_CACHE_PATH = Path(config.ROOT_DIR) / ".cache" / "code_bdong.json"
_BDONG_FETCH_TIMEOUT = 30

_bdong_table: pd.DataFrame | None = None
_bdong_lock = threading.Lock()

# 번지 토큰: '680', '680-63', '680번지'. 주소 문자열에서 지역명 부분과 분리해 뽑는다.
# 주의: "번지?"는 "번"만 필수고 "지"만 선택이 되는 실수라(전체가 선택이어야 함),
# "(?:번지)?"로 묶어야 한다 - 실제로 "2-2"처럼 "번지" 글자가 없는 주소에서
# 매칭이 통째로 실패하는 버그로 나타났다(2026-07-23 실측 테스트로 발견).
_BUNJI_PATTERN = re.compile(r"(\d{1,4})(?:-(\d{1,4}))?(?:번지)?")


def _load_bdong_table() -> pd.DataFrame:
    """전국 법정동코드 표를 최초 호출 시 1회 내려받아 디스크+메모리에 캐싱.
    말소(폐지)된 동은 제외해서 현재 유효한 코드만 남긴다.
    """
    global _bdong_table
    if _bdong_table is not None:
        return _bdong_table
    with _bdong_lock:
        if _bdong_table is not None:
            return _bdong_table
        if _BDONG_CACHE_PATH.exists():
            raw = json.loads(_BDONG_CACHE_PATH.read_text(encoding="utf-8"))
        else:
            resp = requests.get(_BDONG_JSON_URL, timeout=_BDONG_FETCH_TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()
            _BDONG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _BDONG_CACHE_PATH.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        df = pd.DataFrame(raw["data"]).fillna("")
        df = df[df["말소일자"].astype(str).str.strip() == ""].copy()
        code = df["법정동코드"].astype(str)
        df["sigungu_code"] = code.str[:5]
        df["bdong_code"] = code.str[5:]
        _bdong_table = df
        return df


def _find_region_codes(address: str) -> tuple[str, str] | None:
    """주소 텍스트에서 번지 토큰을 제외한 지역명 부분으로 법정동코드 표를 AND 매칭.

    TODO(도로명주소 미지원, 2026-07-23 확인): 이 표는 읍면동명ᆞ동리명(법정동
    체계)만 있고 도로명 컬럼이 없어서, "능동로 87"처럼 도로명 주소를 주면
    매칭이 안 된다. 건축HUB API 자체도 지번(bun/ji) 기준이라 도로명 번호를
    그대로 넣으면 안 됨(같은 건물이 도로명ᆞ지번 번호가 서로 다름 - VWorld
    지오코딩 테스트로 실측 확인). 나중에 지번 대신 도로명 주소로도 찾을 수
    있게 할 예정 - lookup_land_zone의 VWorld 지오코더(도로명 입력에도 법정동
    이름을 같이 돌려줌)를 재사용하는 방향이 유력한 후보.
    """
    df = _load_bdong_table()
    hay = df["시도명"] + " " + df["시군구명"] + " " + df["읍면동명"] + " " + df["동리명"]
    terms = [t for t in re.split(r"[\s,]+", address) if t and not _BUNJI_PATTERN.fullmatch(t)]
    if not terms:
        return None
    mask = pd.Series([True] * len(df), index=df.index)
    for term in terms:
        mask &= hay.str.contains(term, na=False, regex=False)
    matched = df[mask]
    if len(matched) == 0:
        return None
    row = matched.iloc[0]
    return row["sigungu_code"], row["bdong_code"]


def _extract_bunji(address: str) -> tuple[str, str]:
    """공백으로 나눈 토큰 중 번지 패턴과 완전히 일치하는 토큰만 채택한다.
    .search()로 문자열 전체를 훑으면 "성수동2가"처럼 동 이름 안에 포함된
    숫자(2)를 번지로 잘못 집어낸다(2026-07-23 실측 버그: '성수동2가 275-5'가
    번지 '2'로 잘못 추출됨) - _find_region_codes의 fullmatch 방식과 동일하게
    토큰 단위로 정확히 매칭해야 한다.
    """
    for token in re.split(r"[\s,]+", address):
        match = _BUNJI_PATTERN.fullmatch(token)
        if match:
            return match.group(1), match.group(2) or ""
    return "", ""


def _format_date(raw: str) -> str:
    """건축HUB 날짜 필드(YYYYMMDD 문자열)를 YYYY-MM-DD로 변환. 형식이 아니면 원본 그대로."""
    raw = str(raw).strip()
    if re.fullmatch(r"\d{8}", raw):
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return raw or "정보없음"


_ledger_client: BuildingLedger | None = None


def _get_ledger_client() -> BuildingLedger:
    global _ledger_client
    if _ledger_client is None:
        _ledger_client = BuildingLedger(config.ARCHHUB_SERVICE_KEY or "")
    return _ledger_client


def _query_building_ledger(sigungu_code: str, bdong_code: str, bun: str, ji: str) -> pd.DataFrame | None:
    """건축HUB 표제부 API를 직접 REST 호출(PublicDataReader는 URL/컬럼매핑만 재사용).
    결과 없으면 None.
    """
    inst = _get_ledger_client()
    # PublicDataReader가 하드코딩한 BldRgstService_v2는 폐지된 URL이라 항상
    # 500 "Unexpected errors"를 반환한다(2026-07-23 실측: garbage/빈 키로도
    # 동일하게 실패, 다른 data.go.kr API는 정상 401을 반환해 게이트웨이 자체는
    # 정상임을 확인) - 실제 승인ᆞ정상 동작하는 BldRgstHubService를 직접 호출.
    # meta_dict/translate_columns 등 컬럼 매핑은 계속 라이브러리 걸 재사용한다.
    url = f"{config.ARCHHUB_LEDGER_ENDPOINT}/getBrTitleInfo"
    params = {
        "serviceKey": config.ARCHHUB_SERVICE_KEY,
        "sigunguCd": sigungu_code,
        "bjdongCd": bdong_code,
        "numOfRows": 20,
        "pageNo": 1,
        "_type": "json",
    }
    if bun:
        params["bun"] = bun.zfill(4)
    if ji:
        params["ji"] = ji.zfill(4)

    resp = requests.get(url, params=params, timeout=20)  # 정부 API가 가끔 느려서(10초 타임아웃 실측) 여유있게
    resp.raise_for_status()
    data = resp.json()
    body = data.get("response", {}).get("body", {})
    items = body.get("items")
    item = (items.get("item") if isinstance(items, dict) else items) if items else None
    if not item:
        return None
    if isinstance(item, dict):
        item = [item]
    df = pd.DataFrame(item)
    return inst.translate_columns(df)


@tool
def lookup_building_ledger(address: str) -> str:
    """주소(번지 포함)로 건축물대장 표제부(기존 건물의 현재 등록 정보)를 자동 조회합니다.

    - 용도변경ᆞ대수선ᆞ증축처럼 "기존 건물이 있는" 케이스에서만 의미가 있습니다.
      신축(빈 땅에 새로 짓는 경우)에는 대장이 없으니 이 도구를 쓰지 마세요.
    - 조회 성공 시 결과(현재 등록된 용도ᆞ연면적ᆞ층수 등)를 참고해서 record_case_facts를
      채우되, 사용자가 말한 "희망 용도"와 대장상 "기존 용도"를 혼동하지 마세요.
    - 조회 실패나 결과 없음이 오면 사용자에게 직접 물어보세요.
    """
    if not config.ARCHHUB_SERVICE_KEY:
        return "건축물대장 자동 조회가 설정되어 있지 않습니다(API 키 없음). 사용자에게 직접 물어보세요."

    logger.info("[도구] lookup_building_ledger(address=%r)", address)
    codes = _find_region_codes(address)
    if codes is None:
        return f"'{address}' 주소의 법정동 코드를 찾지 못했습니다. 사용자에게 더 정확한 주소(구/동 이름)를 요청하세요."
    sigungu_code, bdong_code = codes

    bun, ji = _extract_bunji(address)
    if not bun:
        return f"'{address}'에서 번지를 찾지 못했습니다. 사용자에게 번지까지 포함한 주소를 요청하세요."

    try:
        df = _query_building_ledger(sigungu_code, bdong_code, bun, ji)
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.warning("[lookup_building_ledger] 조회 실패: %s", exc)
        return "건축물대장 조회에 실패했습니다(서비스 오류). 사용자에게 직접 물어보세요."

    if df is None or len(df) == 0:
        return f"'{address}'에 등록된 건축물대장이 없습니다(신축 예정 대지이거나 미등록 상태일 수 있음)."

    row = df.iloc[0]
    parts = [f"'{address}' 건축물대장 표제부 조회 결과:"]
    if row.get("건물명"):
        parts.append(f"- 건물명: {row['건물명']}")
    parts.append(f"- 주용도: {row.get('주용도코드명', '정보없음')}")
    parts.append(f"- 구조: {row.get('구조코드명', '정보없음')}")
    parts.append(f"- 연면적: {row.get('연면적', '?')}㎡ / 대지면적: {row.get('대지면적', '?')}㎡")
    parts.append(f"- 지상 {row.get('지상층수', '?')}층 / 지하 {row.get('지하층수', '?')}층")
    parts.append(f"- 건폐율: {row.get('건폐율', '?')}% / 용적률: {row.get('용적률', '?')}%")
    parts.append(f"- 사용승인일: {_format_date(row.get('사용승인일', ''))}")
    parts.append("[출처: 국토교통부 건축HUB 건축물대장정보 서비스]")
    return "\n".join(parts)
