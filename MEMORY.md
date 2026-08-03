# MEMORY.md — 이 프로젝트의 의사결정ᆞ패턴ᆞ인사이트 (세컨드 브레인)

> 이 파일은 **인덱스**다. 각 줄이 `docs/memory/<파일>.md`(상세) 또는
> `docs/project_report.md`의 특정 섹션(증거ᆞ숫자ᆞ전체 서사)을 가리킨다.
> 여기 자체에 긴 설명을 쓰지 않는다 — 그러면 곧 CLAUDE.md가 겪었던 것과
> 같은 "규칙과 상세가 뒤섞이는" 문제가 재발한다.
>
> `CLAUDE.md`(규칙ᆞ작업 방식) vs 이 파일(왜 그렇게 결정했는지ᆞ재사용 가능한
> 패턴)의 역할 차이는 `docs/README.md` 참고.

## 아키텍처 원칙

- [[docs/memory/rule-engine-vs-llm]] — 판정은 코드, LLM은 설명(핵심 원칙, 다른 모든 패턴의 근거)
- [[docs/memory/domain-registry-pattern]] — 도메인이 늘어날 때의 배선 패턴(DOMAIN_CONFIGS)
- [[docs/memory/rule-justification-pattern]] — 판정 사유도 코드가 만드는 패턴

## 신뢰성 엔지니어링

- [[docs/memory/guard-node-pattern]] — 프롬프트 한계 → 그래프 레벨 사후 검증
- [[docs/memory/tool-calling-reliability]] — 도구 스킵 원인 규명ᆞtool_choice 강제(해결됨)
- [[docs/memory/external-api-reliability]] — 외부 API 5xx 재시도 패턴

## 성능ᆞ컨텍스트

- [[docs/memory/context-budget-management]] — 토큰 예산 실측ᆞ인덱스 미갱신 발견ᆞ자동 절삭
- [[docs/memory/conditional-prompt-assembly]] — 도메인 블록 조건부 조립(프롬프트 다이어트)

## 콘텐츠 전략

- [[docs/memory/static-vs-rag-content]] — 정적 하드코딩 vs RAG 전환 기준

## 판정 정확성 (법령 해석)

- [[docs/memory/legal-interpretation-cases]] — 건축사 의무ᆞ식품 완제품 판매 등 실제 결정 사례

## 프로세스ᆞ방법론

- [[docs/memory/measurement-methodology]] — 실측 우선 원칙, 반복된 "A로 보였는데 B였다" 패턴

---

**다른 문서와의 관계**: `docs/project_report.md`가 원본 증거(숫자ᆞ표ᆞ
전체 타임라인)이고, 여기 인덱스가 가리키는 `docs/memory/*.md`는 그걸
토픽별로 재배열한 "지금 알아야 할 결론"만 담은 요약이다. "다음 할 일"은
여기 없다 — `docs/phase1_roadmap.md`가 단일 진실.
