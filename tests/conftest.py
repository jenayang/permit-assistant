from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_MARKDOWN_FIXTURE = Path(__file__).parent / "fixtures" / "markdown_render_check.js"


@pytest.fixture(scope="session")
def markdown_render_results() -> dict[str, bool]:
    """tests/fixtures/markdown_render_check.js를 한 번만 실행해서 {label: passed}
    로 돌려준다 - src/static/index.html의 renderMarkdown() 블록을 그대로 잘라내
    평가한 결과다."""
    if shutil.which("node") is None:
        pytest.skip("node가 설치되어 있지 않아 마크다운 렌더링(JS) 테스트를 건너뜁니다.")

    proc = subprocess.run(
        ["node", str(_MARKDOWN_FIXTURE)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        pytest.fail(f"markdown_render_check.js 실행 실패:\n{proc.stderr}")

    results = json.loads(proc.stdout)
    return {item["label"]: item["passed"] for item in results}
