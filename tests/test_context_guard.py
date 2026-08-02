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
    _count_message_chars,
    _fit_context,
    _is_context_overflow_error,
    _is_quota_error,
    estimate_tokens,
)


def _long_history(turns: int, size: int = 4_000) -> list:
    """사람-AI 대화를 turns번 반복한 이력."""
    history = []
    for i in range(turns):
        history.append(HumanMessage(content=f"질문{i} " + "가" * size))
        history.append(AIMessage(content=f"답변{i} " + "나" * size))
    return history


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


def test_normal_conversation_is_not_trimmed():
    """평소 대화(한도의 2~3%)에서는 절삭이 아예 동작하면 안 된다 - 짧은 대화의
    멀쩡한 맥락을 버리지 않는 게 예산 기준 트리거를 쓰는 이유다."""
    prefix = [SystemMessage(content="시스템")]
    history = _long_history(turns=3, size=100)
    assert _fit_context(prefix, history, using_fallback=True) is history


def test_long_conversation_is_trimmed_on_fallback_model():
    """폴백 모델 한도(131,072)에 근접하면 오래된 대화가 잘려야 한다."""
    prefix = [SystemMessage(content="시스템")]
    history = _long_history(turns=20, size=4_000)  # 약 16만 자
    trimmed = _fit_context(prefix, history, using_fallback=True)
    assert len(trimmed) < len(history)
    assert _count_message_chars(trimmed) < _count_message_chars(history)


def test_trimming_keeps_the_most_recent_turn():
    """가장 최근 대화는 반드시 남아야 한다 - 지금 사용자가 방금 한 말이
    잘리면 답변 자체가 불가능하다."""
    prefix = [SystemMessage(content="시스템")]
    history = _long_history(turns=20, size=4_000)
    trimmed = _fit_context(prefix, history, using_fallback=True)
    assert trimmed[-1].content == history[-1].content


def test_same_history_survives_on_primary_model():
    """주 모델(약 100만 토큰)에선 같은 이력이 안 잘려야 한다 - 한도가 모델마다
    달라서, 폴백 기준으로 잘라버리면 Gemini에서 불필요하게 맥락을 잃는다."""
    prefix = [SystemMessage(content="시스템")]
    history = _long_history(turns=20, size=4_000)
    assert _fit_context(prefix, history, using_fallback=False) is history


def test_trimming_does_not_mutate_original_history():
    """절삭은 LLM에 보내는 사본에만 적용되고 원본은 그대로여야 한다 -
    guard_node가 원본 이력을 스캔해 검색ᆞ공사안내 호출 여부를 판정하므로,
    원본을 건드리면 없는 위반을 만들어낸다."""
    prefix = [SystemMessage(content="시스템")]
    history = _long_history(turns=20, size=4_000)
    before = len(history)
    _fit_context(prefix, history, using_fallback=True)
    assert len(history) == before


def test_fallback_limit_is_much_tighter_than_primary():
    """폴백 모델 한도가 주 모델보다 좁다는 전제가 깨지면 경고 시점 계산이
    무의미해진다(예: 모델 교체 시)."""
    assert config.CEREBRAS_CONTEXT_LIMIT < config.GEMINI_CONTEXT_LIMIT
    assert 0 < config.CONTEXT_WARN_RATIO < 1
