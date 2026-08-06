"""Gold Set 자체의 타당성 검증.

Gold Set은 추출 정확도를 재는 **정답지**라, 여기가 틀리면 멀쩡한 동작을 오답으로
집계해 없는 버그를 쫓게 된다. 실제로 그런 일이 있었다(2026-08-02): 간판 케이스에
"돌출간판+floor", "현수막+length_m", "지주이용간판+area_sqm"을 정답으로 뒀는데,
그 필드들은 해당 sign_type의 판정에 아예 안 쓰이는 값이었다. 모델은 도구 설명의
적용 범위 표기를 정확히 따랐는데 정답지 쪽이 틀렸던 것이다.

여기서는 Gold Set의 기대 필드가 (1) 실제 도구 스키마에 있는 필드이고
(2) 간판의 경우 그 sign_type이 실제로 쓰는 필드인지를 코드로 대조한다.
"""
from __future__ import annotations

import pytest

from src.agents import TOOLS
from src.agents.signage import REQUIRED_FIELDS_SIGNAGE
from tests.fixtures.extraction_gold_set import GOLD_SET

_RECORD_TOOL = {
    "permit": "record_case_facts",
    "food": "record_food_facts",
    "fire": "record_fire_facts",
    "signage": "record_signage_facts",
}
_TOOL_FIELDS = {t.name: set(t.args) for t in TOOLS}


@pytest.mark.parametrize("domain,text,expected", GOLD_SET)
def test_expected_fields_exist_in_tool_schema(domain, text, expected):
    """기대 필드가 실제 도구 스키마에 있어야 한다 - 오타나 옛 필드명을 쓰면
    영원히 0% 추출로 잡힌다."""
    valid = _TOOL_FIELDS[_RECORD_TOOL[domain]]
    unknown = set(expected) - valid
    assert not unknown, f"{text!r}: 스키마에 없는 필드 {unknown}"


@pytest.mark.parametrize(
    "text,expected",
    [(t, e) for d, t, e in GOLD_SET if d == "signage"],
)
def test_signage_expected_fields_are_used_by_that_sign_type(text, expected):
    """간판은 sign_type마다 판정에 쓰는 필드가 다르다 - 안 쓰이는 필드를
    정답으로 기대하면 모델이 옳게 행동해도 오답으로 집계된다."""
    sign_type = expected.get("sign_type")
    assert sign_type, f"{text!r}: 간판 케이스엔 sign_type이 있어야 한다"
    allowed = {"sign_type", *REQUIRED_FIELDS_SIGNAGE.get(sign_type, [])}
    unused = set(expected) - allowed
    assert not unused, (
        f"{text!r}: {sign_type} 판정에 안 쓰이는 필드를 기대하고 있다 {unused} "
        f"(실제 사용 필드: {sorted(allowed)})"
    )


def test_gold_set_covers_every_domain():
    """도메인이 늘었는데 Gold Set에 안 들어가면 그 도메인은 측정 사각지대가 된다."""
    assert {d for d, _, _ in GOLD_SET} == set(_RECORD_TOOL)
