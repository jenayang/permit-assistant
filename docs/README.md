# docs 안내 — 어느 문서를 봐야 하나

문서마다 **역할이 하나씩만** 있다. 예전엔 "다음 할 일" 목록이
`phase1_roadmap.md`ᆞ`project_report.md` 5장ᆞ`idea_notes.md` 세 곳에 흩어져 있다가
서로 어긋났고(2026-08-02에 리포트 4장과 5장이 모순되는 상태까지 갔다), 그래서
아래처럼 갈랐다.

| 문서 | 역할 | 언제 보나 |
|---|---|---|
| [`phase1_roadmap.md`](phase1_roadmap.md) | **현재 상태 + 다음 할 일 (단일 진실)** | 뭘 할지 정할 때 |
| [`project_report.md`](project_report.md) | **왜 그렇게 만들었는지** — blocker별 딥다이브ᆞ실측ᆞ선택 이유 | 기존 결정의 배경이 궁금할 때 |
| `dev_log_YYYY-MM-DD.md` | 그날 세션의 스냅샷(시점 기록, 이후 갱신 안 함) | 특정 날짜에 무슨 일이 있었는지 |
| [`idea_notes.md`](idea_notes.md) | 외부 자료 리뷰에서 나온 아이디어(대부분 미착수) | 새 레포ᆞ자료를 검토할 때 |
| [`use_cases.md`](use_cases.md) / [`test_scenarios.md`](test_scenarios.md) | 사용자 사례ᆞ판정 시나리오 | 규칙 엔진을 고칠 때 |
| [`data_sources.md`](data_sources.md) | 법령ᆞ조례 출처 | 데이터를 추가할 때 |
| [`project_portfolio.md`](project_portfolio.md) | 취업용 회고(독자가 다름) | 포트폴리오 정리 |

## 규칙

- **"다음 할 일"은 `phase1_roadmap.md`에만 적는다.** 다른 문서에서는 링크만 건다.
- `project_report.md`는 "무엇을 할지"가 아니라 **"왜 그렇게 했는지"**를 남긴다.
  숫자ᆞ실측ᆞ기각된 가설을 함께 적어야 나중에 같은 실수를 안 반복한다.
- `dev_log_*.md`는 **그 시점의 기록**이라 나중에 사실이 바뀌어도 고치지 않는다.
  대신 최신 상태는 로드맵을 보면 된다.

## 가장 최근 세션

[`dev_log_2026-08-02.md`](dev_log_2026-08-02.md) — 프롬프트 최적화ᆞtool-calling
신뢰성ᆞ인덱스ᆞ컨텍스트. **다른 컴퓨터에서 이어서 작업한다면 그 문서의 "새 환경
세팅"부터 읽을 것**(`.env`ᆞ`chroma_db/`가 git에 없어서 인덱싱을 빠뜨리면 에러 없이
조용히 틀린 답이 나간다).
