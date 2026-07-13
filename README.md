# 🏢 Permit Assistant

> 서울시 건축 인허가 사전진단 AI 어시스턴트

카페 창업, 사무실 리모델링 등 소규모 상업시설의 인허가 절차를 안내하고 사전 진단하는 챗봇입니다. LangGraph 기반 ReAct 에이전트로 상황에 맞는 법령/조례를 검색하고 종합적인 답변을 제공합니다.

## 특징

- **상황 기반 진단**: 사용자 상황에 맞는 인허가 유형 자동 판별
- **정확한 법령 인용**: 건축법, 서울시 조례 기반 답변
- **사전 문제 감지**: 주차, 정화조, 소방 등 놓치기 쉬운 항목 자동 진단
- **절차 로드맵**: 단계별 소요 시간 및 필요 서류 안내

## 프로젝트 로드맵

### Phase 1: MVP (2주)
- [x] 프로젝트 세팅
- [x] 데이터 수집 (건축법, 서울시 조례)
- [x] 카페 창업 시나리오 대응 (신축/용도변경/대수선 3케이스는 실제 질의 검증 진행 중)
- [x] 기본 사전 진단 (주차·정화조·소방·장애인 편의시설·위생 5항목)

### Phase 2: 근린생활시설 확장
- [ ] 사무실, 소매점 등 추가
- [ ] 소방/위생 관련 확장
- [ ] 사전 진단 항목 확대

### Phase 3: 전체 확장
- [ ] 전 지자체 지원
- [ ] 웹 UI (FastAPI + 프론트)
- [ ] 사용자별 프로젝트 관리

## 기술 스택

- **LLM**: Google Gemini 2.5 Flash
- **임베딩**: HuggingFace `paraphrase-multilingual-MiniLM-L12-v2`
- **벡터DB**: ChromaDB
- **에이전트**: LangGraph (ReAct 패턴)
- **API**: FastAPI (Phase 2+)

## 프로젝트 구조

```
permit-assistant/
├── src/                    # 소스 코드
│   ├── config.py           # 환경 설정
│   ├── ingestion.py        # 문서 로딩 + 청킹
│   ├── retriever.py        # 벡터 검색
│   ├── tools.py            # 에이전트 도구
│   ├── agent.py            # LangGraph 에이전트
│   ├── pipeline.py         # CLI 인터페이스
│   └── api.py              # FastAPI 서버 (나중)
├── data/
│   ├── laws/               # 법령 (건축법, 국토법 등)
│   ├── ordinances/         # 서울시 조례
│   ├── procedures/         # 인허가 절차 문서
│   └── samples/            # 테스트 시나리오
├── docs/                   # 로드맵, 사례 문서
├── scripts/                # 유틸 스크립트
├── database/               # ChromaDB (gitignore)
├── pyproject.toml
└── README.md
```

## 시작하기

### 1. 가상환경 + 패키지 설치

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
# .env 열어서 GEMINI_API_KEY 입력
```

### 3. 데이터 준비

`data/` 폴더에 관련 자료 배치:
- `laws/`: 건축법, 시행령, 시행규칙
- `ordinances/`: 서울시 건축조례, 도시계획조례
- `procedures/`: 인허가 절차 안내 문서

### 4. 인덱싱

```bash
python -m src.pipeline --ingest --reset
```

### 5. 사용

```bash
# 단발 질문
python -m src.pipeline --query "강남구 30평 카페 창업 절차는?"

# 대화 모드
python -m src.pipeline --interactive --user jena
```

## 사용 예시

```
You: 강남구에 40평 카페 열려고 해. 기존 사무실 자리인데 뭐 준비해야 해?

Bot: [상황 분석]
     기존 사무실을 카페로 → 근린생활시설 내 용도변경 케이스입니다.
     
     [필요 절차]
     1. 용도변경 신고 [출처: 건축법 제19조]
     2. 소방 사전 협의
     3. 위생 신고 (구청 위생과)
     
     [사전 진단 - 주의사항]
     ⚠️ 정화조 용량: 카페는 사무실보다 오수량 많음
     ⚠️ 주차대수: 근린생활시설 부설주차장 기준 확인
     
     [예상 소요 기간]
     서류 준비 2주 + 심의 3-4주 = 총 5-6주
```

## 작성자

jenyyang · KTB4 Bootcamp