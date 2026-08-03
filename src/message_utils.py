"""메시지 content에서 순수 텍스트를 뽑는 공용 유틸.

agent.py 여러 곳(프롬프트 조립ᆞ컨텍스트 추정ᆞguard 검증)과 pipeline.py가
전부 이 함수를 가져다 쓴다 - 각자 따로 구현하면 처리 방식이 어긋나는 버그가
실제로 있었다(Gemini의 content가 단순 문자열이 아니라 블록 리스트로 오는
경우를 한쪽만 놓침).
"""
from __future__ import annotations


def extract_text(content) -> str:
    """AIMessage.content에서 순수 텍스트만 뽑는다. 최신 Gemini는 content가
    단순 문자열이 아니라 [{"type": "text", "text": "..."}] 같은 블록 리스트로
    올 수 있어서, 이 함수 하나로 통일해 agent.py/pipeline.py가 서로 다르게
    처리하다 어긋나는 걸 막는다.
    """
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return content or ""
