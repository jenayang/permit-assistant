# 판정은 규칙 엔진, LLM은 설명

**원칙**: 허가/신고 여부 등 법령상 객관적으로 정해지는 값은 파이썬 함수
(`classify_case` 등 4종)가 결정론적으로 계산한다. LLM은 결과를 설명만 한다.

**왜**: LLM에 맡겼더니 계산을 틀리는 사례가 실제로 있었다. 법적 판정은
안전-critical해서 확률적 오류를 허용 못 함. 이후 `guard_node`ᆞ`tool_choice`
강제도 전부 같은 논리("프롬프트로 안 되면 코드로 뺀다")의 확장이다.

**적용**: 판정 규칙(법령 해석) 변경은 **사용자 결정 사항** — 혼자 고치지 말고
근거(조문 원문)를 제시하고 물어볼 것. 사례: [[legal-interpretation-cases]]

**관련**: [[domain-registry-pattern]] · [[rule-justification-pattern]] ·
[[guard-node-pattern]]

상세: `docs/project_report.md` 4-1
