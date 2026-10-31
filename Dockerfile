FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 의존성 레이어 먼저 캐싱 (코드만 바뀌면 재설치 안 함)
# README.md도 필요 (pyproject.toml의 readme 필드를 hatchling이 빌드 시 읽음)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen

# 코드 + 법령 원문
COPY src/ src/
COPY data/ data/

# 빌드 시점에 벡터DB 미리 구축 (컨테이너 시작을 빠르게).
# src/agent.py가 import 시점에 ChatGoogleGenerativeAI를 즉시 생성하므로
# (실제 ingest 로직은 LLM을 안 쓰지만) 키 "존재"는 필요 - 빌드 전용 더미값.
# 진짜 키는 이미지에 안 굽고 `docker run -e`로 실행 시점에 넣는다.
ARG GEMINI_API_KEY=build-time-dummy-key
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
RUN uv run python -m src.pipeline --ingest

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
