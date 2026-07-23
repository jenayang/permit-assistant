# 🏢 Permit Assistant

> 서울시 건축 인허가 사전진단 AI 어시스턴트

신축ᆞ증축ᆞ용도변경ᆞ대수선 등 건축 인허가 절차를 안내하고 사전 진단하는 챗봇입니다. LangGraph 기반 ReAct 에이전트로 상황에 맞는 법령/조례를 검색하고 종합적인 답변을 제공합니다.

## 특징

- **상황 기반 진단**: 사용자 상황에 맞는 인허가 유형 자동 판별
- **정확한 법령 인용**: 건축법, 서울시 조례 기반 답변
- **사전 문제 감지**: 주차, 정화조, 소방 등 놓치기 쉬운 항목 자동 진단
- **법령 구조 인식 청킹**: 본문(조항)/부칙/별표를 구분해서 별표 안 표 데이터가 가짜 조항으로 섞이는 걸 방지, 조항 원문은 부모-자식 청킹으로 검색은 짧게・답변은 잘리지 않게
- **절차 로드맵**: 상황분석/필요절차/필수서류/사전진단/예상기간을 구조화해서 안내
- **LLM 폴백**: Gemini 무료 티어 할당량 소진 시 Cerebras로 자동 전환
- **증분 인덱싱**: 문서 변경분만 해시 비교 후 재임베딩 (전체 재인덱싱 불필요)
- **웹 채팅 UI**: 브라우저에서 바로 사용 가능 (`/ui`)

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
- [x] 웹 UI (FastAPI + 정적 프론트엔드)
- [ ] 사용자별 프로젝트 관리

## 기술 스택

- **LLM**: Google Gemini 2.5 Flash Lite (기본) → Cerebras(`gemma-4-31b` 등, 할당량 소진 시 자동 폴백)
- **임베딩**: HuggingFace `intfloat/multilingual-e5-base` (로컬 CPU 추론)
- **벡터DB**: ChromaDB (`chroma_db/`, 로컬 영구 저장)
- **증분 인덱싱**: LangChain `SQLRecordManager` (콘텐츠 해시 기반, 변경분만 재임베딩)
- **에이전트**: LangGraph (ReAct 패턴)
- **API / UI**: FastAPI + 정적 HTML/JS 채팅 UI

## 프로젝트 구조

```
permit-assistant/
├── src/                    # 소스 코드
│   ├── config.py           # 환경 설정
│   ├── ingestion.py        # 문서 로딩 + 청킹
│   ├── law_chunker.py      # 법령 조항 단위 청킹 (본문/부칙/별표 구역 분리)
│   ├── parent_store.py     # 조항 원문 저장소 (부모-자식 청킹의 부모)
│   ├── retriever.py        # 벡터 검색 + 증분 인덱싱
│   ├── tools.py            # 에이전트 도구
│   ├── agent.py            # LangGraph 에이전트 (Gemini/Cerebras 폴백)
│   ├── pipeline.py         # CLI 인터페이스
│   ├── api.py              # FastAPI 서버
│   ├── evaluate.py         # 응답 품질 평가
│   └── static/             # 웹 채팅 UI (/ui)
├── data/
│   ├── laws/               # 법령 (건축법, 국토법 등)
│   ├── ordinances/         # 서울시 조례
│   ├── procedures/         # 인허가 절차 문서
│   └── samples/            # 테스트 시나리오
├── docs/                   # 로드맵, 사례 문서
├── tests/
├── chroma_db/               # ChromaDB + 인덱싱 해시 기록 (gitignore)
├── pyproject.toml
└── README.md
```

## 시작하기

### 1. 패키지 설치 ([uv](https://docs.astral.sh/uv/) 사용)

```bash
uv sync
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
```

`.env`에 다음 값을 입력:
- `GEMINI_API_KEY` (필수) — [발급](https://aistudio.google.com/apikey)
- `CEREBRAS_API_KEY` / `CEREBRAS_MODEL` (선택) — Gemini 무료 티어 할당량(하루 20회) 소진 시 자동 폴백용

### 3. 데이터 준비

`data/` 폴더에 관련 자료 배치 (지원 포맷: pdf, md, txt, html, hwp):
- `laws/`: 건축법, 시행령, 시행규칙
- `ordinances/`: 서울시 건축조례, 도시계획조례
- `procedures/`: 인허가 절차 안내 문서

법령 원문 PDF에 별표(첨부 표)가 이미지로 박혀 텍스트 추출이 안 되는 경우,
국가법령정보센터에서 해당 별표만 한글(hwp) 파일로 따로 받아 같은 폴더에
넣으면 자동으로 인덱싱된다(내부적으로 `hwp5html`로 표까지 변환).

### 4. 인덱싱

```bash
# 최초 인덱싱 (또는 완전히 새로 구축하고 싶을 때)
uv run python -m src.pipeline --ingest --reset

# 이후 데이터 추가/수정 시 (안 바뀐 문서는 재임베딩 스킵)
uv run python -m src.pipeline --ingest
```

### 5. 사용

**CLI**
```bash
# 단발 질문
uv run python -m src.pipeline --query "강남구 30평 카페 창업 절차는?"

# 대화 모드
uv run python -m src.pipeline --interactive --user jena
```

**API 서버 + 웹 UI**
```bash
uv run uvicorn src.api:app --reload
```
- 웹 채팅 UI: http://localhost:8000/ui
- API 문서: http://localhost:8000/docs

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

jenayang · KTB4 Bootcamp
