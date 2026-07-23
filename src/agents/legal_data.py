"""법정 임계값(permit_thresholds.yaml) 로더 - permit.py/site.py가 공유.

Chroma 벡터스토어처럼 초기화 비용이 큰 리소스가 아니라 그냥 작은 YAML 파일을
동기로 읽는 것뿐이라, retriever.py의 스레드 락 있는 싱글턴 패턴까지는
필요 없다 - 락 없는 lazy 캐시로 충분하다.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_THRESHOLDS_PATH = Path(__file__).parent / "permit_thresholds.yaml"
_thresholds: dict | None = None


def get_thresholds() -> dict:
    """permit_thresholds.yaml을 최초 호출 시 1회 로드해서 캐싱."""
    global _thresholds
    if _thresholds is None:
        _thresholds = yaml.safe_load(_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    return _thresholds
