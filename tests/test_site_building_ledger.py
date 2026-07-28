"""lookup_building_ledger의 주소 해석 경로(VWorld 기본 경로 + 법정동표
폴백) 테스트. 실제 VWorld/건축HUB API는 호출하지 않는다 - 네트워크ᆞAPI
키에 의존하지 않고 항상 결정론적으로 돌게 하기 위해 monkeypatch로 경계만
검증한다.

2026-07-28: 이 프로젝트가 "도로명주소 미지원"이었던 지점을 VWorld
지오코딩+역지오코딩으로 해결하면서(site.py의 _vworld_resolve_region_and_bunji),
기존 전국 법정동표 텍스트 매칭(_find_region_codes/_extract_bunji)은 지우지
않고 VWorld 장애 시 자동 폴백으로 남겼다 - 그 자동 전환이 실제로 동작하는지
가 이 파일의 핵심 검증 대상이다.
"""
from __future__ import annotations

import pandas as pd

from src.agents import site


def test_resolve_parses_road_and_parcel_address_identically(monkeypatch):
    """VWorld 응답의 level4LC(법정동코드)ᆞlevel5(지번)를 정확히 쪼개는지 -
    실제 응답 구조는 2026-07-28에 "능동로 87"/"자양동 2-2" 둘 다로 실측
    확인했다(광진구=11215, 자양동=10500, 번지 2-2)."""
    monkeypatch.setattr(site.config, "VWORLD_API_KEY", "dummy-key")
    monkeypatch.setattr(site, "_vworld_geocode", lambda address: (127.07, 37.54))
    monkeypatch.setattr(
        site, "_vworld_query_parcel_address",
        lambda x, y: {"level4LC": "1121510500", "level5": "2-2"},
    )

    result = site._vworld_resolve_region_and_bunji("아무 주소나(모킹됨)")
    assert result == ("11215", "10500", "2", "2")


def test_resolve_handles_bunji_without_dash(monkeypatch):
    """지번에 부번이 없는 경우("87"처럼 '-' 없음) - ji는 빈 문자열이어야 한다."""
    monkeypatch.setattr(site.config, "VWORLD_API_KEY", "dummy-key")
    monkeypatch.setattr(site, "_vworld_geocode", lambda address: (127.07, 37.54))
    monkeypatch.setattr(
        site, "_vworld_query_parcel_address",
        lambda x, y: {"level4LC": "1121510500", "level5": "87"},
    )

    result = site._vworld_resolve_region_and_bunji("아무 주소나(모킹됨)")
    assert result == ("11215", "10500", "87", "")


def test_resolve_returns_none_without_api_key(monkeypatch):
    monkeypatch.setattr(site.config, "VWORLD_API_KEY", "")
    assert site._vworld_resolve_region_and_bunji("아무 주소") is None


def test_resolve_returns_none_when_geocode_fails(monkeypatch):
    monkeypatch.setattr(site.config, "VWORLD_API_KEY", "dummy-key")
    monkeypatch.setattr(site, "_vworld_geocode", lambda address: None)
    assert site._vworld_resolve_region_and_bunji("존재하지 않는 주소") is None


def test_resolve_returns_none_when_bdong_code_malformed(monkeypatch):
    """level4LC가 10자리가 아니면(예: 도로명 코드만 오고 법정동코드가 비어있는
    경우) 잘못된 코드로 건축HUB를 호출하지 않도록 None을 반환해야 한다."""
    monkeypatch.setattr(site.config, "VWORLD_API_KEY", "dummy-key")
    monkeypatch.setattr(site, "_vworld_geocode", lambda address: (127.07, 37.54))
    monkeypatch.setattr(
        site, "_vworld_query_parcel_address",
        lambda x, y: {"level4LC": "", "level5": "2-2"},
    )
    assert site._vworld_resolve_region_and_bunji("아무 주소") is None


def test_lookup_building_ledger_falls_back_when_vworld_path_fails(monkeypatch):
    """VWorld 경로(_vworld_resolve_region_and_bunji)가 실패하면
    lookup_building_ledger가 자동으로 법정동표 폴백(_find_region_codes/
    _extract_bunji)으로 넘어가는지 - 이번 세션에서 추가한 자동 전환의
    핵심 동작."""
    monkeypatch.setattr(site.config, "ARCHHUB_SERVICE_KEY", "dummy-key")
    monkeypatch.setattr(site, "_vworld_resolve_region_and_bunji", lambda address: None)

    fallback_calls = []

    def fake_find_region_codes(address):
        fallback_calls.append(("region", address))
        return ("11215", "10500")

    def fake_extract_bunji(address):
        fallback_calls.append(("bunji", address))
        return ("2", "2")

    monkeypatch.setattr(site, "_find_region_codes", fake_find_region_codes)
    monkeypatch.setattr(site, "_extract_bunji", fake_extract_bunji)

    captured_args = {}

    def fake_query_building_ledger(sigungu_code, bdong_code, bun, ji):
        captured_args.update(sigungu_code=sigungu_code, bdong_code=bdong_code, bun=bun, ji=ji)
        return pd.DataFrame([{
            "건물명": "테스트빌딩", "주용도코드명": "제1종근린생활시설",
            "구조코드명": "철근콘크리트구조", "연면적": "100", "대지면적": "50",
            "지상층수": "3", "지하층수": "0", "건폐율": "50", "용적률": "150",
            "사용승인일": "20200101",
        }])

    monkeypatch.setattr(site, "_query_building_ledger", fake_query_building_ledger)

    result = site.lookup_building_ledger.func("서울특별시 광진구 자양동 2-2")

    assert fallback_calls, "VWorld 경로 실패 시 폴백 함수가 호출돼야 한다"
    assert captured_args == {"sigungu_code": "11215", "bdong_code": "10500", "bun": "2", "ji": "2"}
    assert "테스트빌딩" in result


def test_lookup_building_ledger_uses_vworld_path_when_available(monkeypatch):
    """VWorld 경로가 성공하면 법정동표 폴백 함수는 아예 호출되지 않아야
    한다(불필요한 호출 없이 바로 VWorld 결과를 쓴다)."""
    monkeypatch.setattr(site.config, "ARCHHUB_SERVICE_KEY", "dummy-key")
    monkeypatch.setattr(
        site, "_vworld_resolve_region_and_bunji",
        lambda address: ("11215", "10500", "2", "2"),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("VWorld 경로가 성공했는데 폴백 함수가 호출됐다")

    monkeypatch.setattr(site, "_find_region_codes", fail_if_called)
    monkeypatch.setattr(site, "_extract_bunji", fail_if_called)
    monkeypatch.setattr(
        site, "_query_building_ledger",
        lambda sigungu_code, bdong_code, bun, ji: pd.DataFrame([{
            "건물명": "능동로빌딩", "주용도코드명": "업무시설",
            "구조코드명": "철골구조", "연면적": "200", "대지면적": "100",
            "지상층수": "5", "지하층수": "1", "건폐율": "60", "용적률": "200",
            "사용승인일": "20220101",
        }]),
    )

    result = site.lookup_building_ledger.func("서울특별시 광진구 능동로 87")
    assert "능동로빌딩" in result
