# ADR 목록 — 트러블슈팅ᆞ아키텍처 결정 기록

[ADR(Architecture Decision Record)](https://github.com/100-hours-a-week/KTB4-AX/tree/main/wiggle/adr)
양식으로, 이 프로젝트에서 실제로 있었던 문제ᆞ검토한 대안ᆞ선택 이유ᆞ
트레이드오프를 기록한다. 전부 **소급 작성**이며, `docs/project_report.md`의
4장(주요 Blocker 딥다이브)과 git 커밋 이력을 근거로 재구성했다 — 자세한
실측 수치ᆞ원본 서사는 `docs/project_report.md`를 함께 참고.

| ADR | 제목 | 날짜 |
|---|---|---|
| [0001](0001-permit-classification-rule-engine-vs-llm.md) | 허가ᆞ신고 판정은 LLM이 아니라 규칙 엔진이 결정론적으로 계산 | 2026-07-22 |
| [0002](0002-conversation-state-sqlite-checkpointer.md) | 대화ᆞ판정 상태를 SQLite 체크포인터로 영구 저장 | 2026-07-24 |
| [0003](0003-guard-node-post-hoc-validation.md) | 프롬프트 지시만으로 부족해 `guard_node`로 답변을 사후 검증 | 2026-07-26 |
| [0004](0004-domain-configs-registry-refactor.md) | 도메인 3개가 되는 시점에 `DOMAIN_CONFIGS` 레지스트리로 리팩터 | 2026-07-27 |
| [0005](0005-static-content-vs-rag-transition.md) | 정적 콘텐츠를 검증 없이 먼저 바꾸지 않고 RAG 전환 전 비교 검증 | 2026-07-28 |
| [0006](0006-rag-tool-skip-guard-check.md) | 판정 시점에만 검색이 몰리는 문제를 guard 체크로 잡음 | 2026-07-29 |
| [0007](0007-external-api-retry-on-5xx.md) | 외부 정부 API는 5xx에 한해서만 짧게 재시도 | 2026-07-29 |
| [0008](0008-act-type-reason-generalization.md) | "왜?" 설명을 용도변경 하나에서 전체 act_type으로 일반화 | 2026-07-29 |
| [0009](0009-tool-calling-degradation-trimming-vs-forced-choice.md) | 컨텍스트 트리밍 구현 후 반증ᆞ롤백, `tool_choice` 강제로 전환 | 2026-07-30~31 |
| [0010](0010-context-budget-audit-prompt-diet.md) | 시스템 프롬프트 토큰 실측, 인시던트 서사ᆞ중복 설명 제거 | 2026-07-31~08-01 |
| [0011](0011-conditional-prompt-assembly.md) | 시스템 프롬프트를 도메인별 블록으로 쪼개 상태에 따라 on/off | 2026-08-01~02 |
| [0012](0012-tool-calling-skip-root-cause-forced-choice-generalization.md) | tool-calling 스킵 원인 규명(도구 선택 문제) + 강제 전 도메인 일반화 | 2026-08-02 |
| [0013](0013-stale-index-detection-and-prevention.md) | "컨텍스트 문제"의 실제 원인은 인덱스 미갱신 — 시작 시 경고로 재발 방지 | 2026-08-02 |
| [0014](0014-context-overflow-observability-and-safety-net.md) | 컨텍스트 초과를 큰 설계 없이 관측+안전망만으로 먼저 대응 | 2026-08-02 |
| [0015](0015-food-business-classification-fix-and-rule-justification.md) | 식품 "완제품만 판매" 오판정 수정 + Rule Justification 패턴 | 2026-08-03 |

## 읽는 순서 추천

- **판정을 왜 LLM에 안 맡기는지** 궁금하면: 0001 → 0003 → 0009 → 0012
  (같은 원칙이 네 번 다른 형태로 반복ᆞ강화되는 과정).
- **컨텍스트ᆞ토큰 관리**가 궁금하면: 0010 → 0011 → 0013 → 0014 (실측 기반
  최적화가 "A로 보였는데 실은 B였다"를 세 번 거치며 좁혀지는 과정).
- **법령 해석 판단**이 궁금하면: 0015 (가장 구체적인 사례).
