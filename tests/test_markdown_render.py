"""src/static/index.html의 renderMarkdown() 번호 목록 렌더링 검증.

순수 JS 렌더 로직이라 tests/fixtures/markdown_render_check.js가 index.html에서
실제 코드 블록을 그대로 잘라내 평가한 결과를 subprocess로 받아온다
(conftest.py의 markdown_render_results fixture).

회귀 배경(2026-08-04, 세션 3d2f20a1): 번호 목록(1. 2. 3.) 중간에 하위 불릿(-)이
끼면 <ol>이 한 번 닫혔다 새로 열리면서, 번호를 브라우저 자동 매김에 맡기던 예전
코드는 다시 1부터 세어 "1. 2. 1."처럼 보였다. <li value="N">으로 원문 번호를
명시하도록 고쳤다.
"""
from __future__ import annotations


def test_ordered_list_survives_bullet_interruption(markdown_render_results):
    assert markdown_render_results["번호 목록이 하위 불릿으로 끊겨도 원문 번호(1,2,3) 그대로 유지"]


def test_ordered_list_without_interruption(markdown_render_results):
    assert markdown_render_results["끊김 없는 번호 목록도 정상 렌더"]


def test_ordered_list_survives_blank_line_interruption(markdown_render_results):
    assert markdown_render_results["빈 줄로 끊긴 번호 목록도 원문 번호 유지"]
