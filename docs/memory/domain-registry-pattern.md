# 도메인 레지스트리 패턴 (DOMAIN_CONFIGS)

**원칙**: 판정 도메인이 늘 때마다 그래프에 if문을 추가하지 않고,
`DOMAIN_CONFIGS`(`agent.py`)에 `_DomainConfig` 항목 하나만 추가한다.

**왜**: 도메인 2~3개까진 if문으로 버텼지만 계속 늘면 배선이 꼬인다. 단,
**인프라를 먼저 일반화하지 않고 3번째 도메인에서 리팩터** — 추측성 일반화를
피하는 이 프로젝트의 원칙을 실제로 적용한 사례.

**필드**: `facts_key`(상태 채널) · `state_key`(결과 저장 위치, permit은
`None`) · `classify_fn`(순수 판정 함수) · `record_tool`(기록 도구명,
tool_choice 강제 대상) · `keywords`(프롬프트 블록 점등 겸 강제 트리거) ·
`domain`(로그명).

**검증됨**: 4번째 도메인(간판) 추가는 목록 한 줄로 끝남. 이후
[[tool-calling-reliability]]의 tool_choice 강제 일반화도 이 필드들만
추가해서 얹혔다 — 패턴이 확장에 강함.

**새 도메인 추가 시**: `src/agents/CLAUDE.md` 체크리스트 참고.

상세: `docs/project_report.md` 4-4
