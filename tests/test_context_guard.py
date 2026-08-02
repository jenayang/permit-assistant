"""컨텍스트 예산 추정ᆞ초과 감지 회귀 테스트.

LLM을 호출하지 않는다 - 추정은 문자 수 기반이고 초과 판별은 에러 문자열
패턴이라 결정론적으로 검증된다.

배경(2026-08-02): 로드맵을 완주하는 대화 하나가 Cerebras 폴백 한도(131,072)의
70%까지 찬다. 트리밍ᆞ요약 같은 큰 설계를 도입하기엔 근거가 약하다고 보고
미뤘지만, 초과 시 프로바이더 원본 에러가 그대로 500으로 나가는 상태였다.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src import config
from src.agent import (
    ContextOverflowError,
    _is_context_overflow_error,
    _is_quota_error,
    estimate_tokens,
)


def test_estimate_grows_with_message_volume():
    """메시지가 늘면 추정치도 늘어야 한다."""
    small = estimate_tokens([HumanMessage(content="안녕하세요")])
    large = estimate_tokens([HumanMessage(content="안녕하세요" * 1000)])
    assert large > small


def test_estimate_includes_tool_schema_overhead():
    """도구 스키마는 messages에 없지만 매 호출 전송되므로 포함돼야 한다 -
    빼먹으면 실제보다 1만 토큰 가까이 과소평가된다."""
    assert estimate_tokens([]) > 5_000


def test_estimate_counts_tool_call_args():
    """tool_calls의 args도 컨텍스트를 차지한다."""
    plain = AIMessage(content="")
    with_args = AIMessage(
        content="",
        tool_calls=[{"name": "record_case_facts", "args": {"note": "x" * 500}, "id": "t1"}],
    )
    assert estimate_tokens([with_args]) > estimate_tokens([plain]) + 400


def test_estimate_is_conservative_for_korean_text():
    """한국어는 실측 1.02 문자/토큰이라 문자 수를 그대로 토큰으로 보면 약간
    과대추정된다 - 가드 입장에선 안전한 방향이고, 과소추정이면 안 된다."""
    text = "성수동에 연면적 120제곱미터 사무실을 카페로 용도변경하려고 합니다." * 20
    base = estimate_tokens([])
    est = estimate_tokens([SystemMessage(content=text)]) - base
    assert est >= len(text) * 0.95


def test_detects_provider_context_errors():
    """프로바이더마다 문구가 달라도 초과로 인식해야 한다."""
    for msg in (
        "400 The input token count exceeds the maximum context length",
        "Request too large for model",
        "This model's maximum context length is 131072 tokens",
        "Please reduce the length of the messages",
    ):
        assert _is_context_overflow_error(Exception(msg)), msg


def test_does_not_confuse_quota_error_with_context_error():
    """할당량 초과와 컨텍스트 초과는 대응이 완전히 다르다(전자는 폴백 전환,
    후자는 새 대화 안내) - 서로 오인하면 안 된다."""
    quota = Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
    assert _is_quota_error(quota)
    assert not _is_context_overflow_error(quota)


def test_context_overflow_error_is_distinct_type():
    """api.py가 이 타입만 골라 413 + 행동 가능한 안내로 바꾼다."""
    assert issubclass(ContextOverflowError, RuntimeError)


def test_fallback_limit_is_much_tighter_than_primary():
    """폴백 모델 한도가 주 모델보다 좁다는 전제가 깨지면 경고 시점 계산이
    무의미해진다(예: 모델 교체 시)."""
    assert config.CEREBRAS_CONTEXT_LIMIT < config.GEMINI_CONTEXT_LIMIT
    assert 0 < config.CONTEXT_WARN_RATIO < 1
