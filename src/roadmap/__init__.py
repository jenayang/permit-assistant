"""로드맵 단계 추적 패키지. 실제 구현은 stage.py에 있고, 여기서 재노출해서
`from src.roadmap import permit_phase_directive` 같은 기존 import 경로를
그대로 유지한다(파일이 패키지로 바뀌어도 외부 소비자는 코드를 안 고쳐도 됨)."""
from src.roadmap.stage import (
    _STAGE_DOMAIN,
    _STAGE_MARKER_LINE_PATTERN,
    _STEP2_KEYWORDS,
    _max_allowed_stage,
    _mentioned_stages,
    _roadmap_status_summary,
    _should_force_construction_guide,
    permit_phase_directive,
)

__all__ = [
    "_STAGE_DOMAIN",
    "_STAGE_MARKER_LINE_PATTERN",
    "_STEP2_KEYWORDS",
    "_max_allowed_stage",
    "_mentioned_stages",
    "_roadmap_status_summary",
    "_should_force_construction_guide",
    "permit_phase_directive",
]
