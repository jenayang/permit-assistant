"""RAGAS를 활용한 RAG 파이프라인 평가.

4가지 지표를 측정합니다:
- faithfulness: 답변이 검색된 문서에 충실한가 (환각 검출)
- answer_relevancy: 답변이 질문과 관련 있는가
- context_precision: 검색된 문서가 정답에 필요한가
- context_recall: 정답에 필요한 정보를 빠짐없이 검색했는가

실행:
    python -m src.evaluate
"""
from __future__ import annotations

import logging
import sys
import types
from datetime import datetime

import pandas as pd
from datasets import Dataset
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# ragas가 langchain_community.chat_models.vertexai를 무조건 import하는데,
# 최신 langchain-community에서는 이 모듈이 완전히 제거됨 (langchain-google-vertexai로 이전).
# 이 프로젝트는 Vertex AI를 쓰지 않으므로 최소 스텁으로 import 에러만 회피.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _ChatVertexAIStub:
        pass

    _vertexai_stub.ChatVertexAI = _ChatVertexAIStub
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src import config
from src.pipeline import query

logger = logging.getLogger(__name__)


# === 평가 데이터셋 ===
# docs/use_cases.md의 주요 시나리오 기반 (Phase 1 완료조건: 카페 창업 질문 10개).
# 실제 프로젝트에서는 별도 yaml/json 파일로 분리하는 게 좋음.
EVAL_DATASET: list[dict[str, str]] = [
    {
        "question": "건폐율의 정의가 뭐야?",
        "ground_truth": (
            "건폐율은 대지면적에 대한 건축면적(대지에 건축물이 둘 이상 있는 경우 "
            "이들 건축면적의 합계)의 비율을 말합니다. 최대한도는 「국토의 계획 및 "
            "이용에 관한 법률」 제77조에 따른 기준을 따르되, 건축법에서 별도로 "
            "완화·강화 규정이 있으면 그에 따릅니다. [출처: 건축법 제55조]"
        ),
    },
    {
        "question": "강남구에 20평 카페 열려고 해. 신축 건물 1층인데 뭐 준비해야 해?",
        "ground_truth": (
            "신축 건물의 근린생활시설(카페)로, 연면적 20평(약 66㎡)은 100㎡ 미만이라 "
            "건축신고 대상입니다. 건축신고(관할 구청 건축과), 위생 신고(관할 구청 "
            "위생과), 사업자등록(세무서)이 필요합니다. 사전 진단 항목으로 주차대수, "
            "소방시설, 정화조 용량, 배기시설, 도로 접도 요건(4m 이상)을 확인해야 합니다."
        ),
    },
    {
        "question": "기존 사무실 자리를 카페로 바꾸려고 하는데 뭘 확인해야 해?",
        "ground_truth": (
            "용도변경 케이스입니다. 기존 시설군과 카페(제2종 근린생활시설)가 동일 "
            "시설군인지 확인하고, 동일하면 건축물대장 기재변경만으로 가능한지 판단합니다. "
            "소방 사전 협의가 필요하고, 인테리어 공사 시 구조 변경 여부도 확인해야 합니다."
        ),
    },
    {
        "question": "카페 리모델링하면서 내력벽을 철거해야 하는데 대수선 허가가 필요해?",
        "ground_truth": (
            "내력벽 철거는 대수선에 해당할 가능성이 높습니다. 대수선 대상인지 먼저 "
            "판별하고, 규모에 따라 대수선 신고 또는 허가 대상인지 구분해야 합니다. "
            "구조 변경이므로 구조 안전 검토가 필요할 수 있습니다."
        ),
    },
    {
        "question": "근린생활시설 카페 창업 시 주차대수는 어떻게 정해져?",
        "ground_truth": (
            "근린생활시설의 부설주차장 설치 기준에 따라 시설 면적 기준으로 주차대수가 "
            "산정됩니다. 서울시 주차장 설치 및 관리 조례에 따른 구체적인 기준을 확인해야 합니다."
        ),
    },
]


def collect_evaluation_data(dataset: list[dict[str, str]]) -> Dataset:
    """파이프라인을 실제로 돌려서 RAGAS 입력 데이터셋 생성."""
    rows = {
        "question": [],  # 평가 쿼리(질문)
        "answer": [],    # 내 모델(파이프라인)에 돌렸을때 결과
        "contexts": [],  # 
        "ground_truth": [],  # 평가 정답
    }

    for i, item in enumerate(dataset, 1):
        logger.info("[%d/%d] %s", i, len(dataset), item["question"])
        result = query(item["question"]) # query 함수가 dict 반환(내 모델에 돌려봄)

        rows["question"].append(item["question"]) # 평가 데이터
        rows["answer"].append(result["answer"])
        rows["contexts"].append(result["contexts"])
        rows["ground_truth"].append(item["ground_truth"]) # 평가 데이터

    return Dataset.from_dict(rows)   # huggingface Dataset 객체 생성 (RAGAS가 이 형식을 요구).


def run_evaluation() -> pd.DataFrame:
    """전체 평가 실행."""
    config.validate() # 평가에 필요한 설정 검증(모델 API 키 + 폴더 확인)

    logger.info("=== 1단계: 파이프라인 실행하여 응답 수집 ===")
    eval_dataset = collect_evaluation_data(EVAL_DATASET)

    logger.info("=== 2단계: RAGAS 평가 (Gemini를 judge로 사용) ===")
    # RAGAS는 내부적으로 LLM과 임베딩 모델을 사용함
    # Gemini로 통일
    judge_llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0,   # ← 채점은 일관성 위해 0
    )
    judge_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=config.GEMINI_API_KEY,
    )
    # 3단계 : RAGAS에게 평가 위임
    result = ragas_evaluate(      # RAGAS 라이브러리가 4개 지표(metrics)를 자동 계산
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = result.to_pandas()
    return df


def save_results(df: pd.DataFrame) -> str:
    """결과를 CSV로 저장."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = config.EVAL_DIR / f"ragas_eval_{timestamp}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return str(output_path)


def print_summary(df: pd.DataFrame) -> None:
    """주요 지표 평균 출력."""
    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    print("\n" + "=" * 60)
    print("RAGAS 평가 결과 요약")
    print("=" * 60)
    for m in metrics:
        if m in df.columns:
            print(f"  {m:25s}: {df[m].mean():.4f}")
    print("=" * 60)


def main() -> None:
    config.setup_logging()   # 1) 로깅 셋팅 (평가 과정에서 로그 출력)
    df = run_evaluation()    # 2) 평가 실행 : RAGPipeline으로 답변 수집 + RAGAS로 지표 계산 → DataFrame 반환
    path = save_results(df)  # 3) 결과 저장 : DataFrame → CSV 파일 저장 + 경로 반환
    print_summary(df)        # 4) 결과 요약 출력 : DataFrame에서 주요 지표 평균 계산하여 출력
    print(f"\n상세 결과 저장: {path}\n")  # CSV 파일 경로 출력


if __name__ == "__main__": # 평가 실행
    main()


# [INFO] === 1단계: 파이프라인 실행하여 응답 수집 ===
# [INFO] [1/3] Adapterz는 어떤 서비스인가요?
# [INFO] [2/3] Startupcode는 어떤 회사인가요?
# [INFO] [3/3] RAG는 어떻게 동작하나요?
# [INFO] === 2단계: RAGAS 평가 (Gemini를 judge로 사용) ===
# [RAGAS가 채점 진행... 시간 좀 걸림]

# ============================================================
# RAGAS 평가 결과 요약
# ============================================================
#   faithfulness             : 0.8542
#   answer_relevancy         : 0.9123
#   context_precision        : 0.7891
#   context_recall           : 0.8234
# ============================================================

# 상세 결과 저장: /path/to/eval_results/ragas_eval_20260703_143020.csv