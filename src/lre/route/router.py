"""Route a holding to AUTO, REVIEW or CONTRADICTION (report 5.3.8, Figure 6).

* CONTRADICTION: a high / medium severity flag -> the whole holding goes to the revenue officer;
* REVIEW: any field below its confidence threshold, a missing required field, an unlinked holding,
  or a low-severity flag (e.g. R4 vocabulary) -> only the failing fields go to a reviewer;
* AUTO otherwise (a 2 % random audit sample of AUTO holdings is still reviewed).

The per-field thresholds come from calibration in Phase 5; until then ``needs_review`` is
decided by the extractor (the baseline uses Tesseract word confidence < 60).
"""
from __future__ import annotations

from typing import Optional

from ..schema import Flag, LinkResult, Route
from ..validate.engine import RuleSet, is_contradiction


def decide_route(needs_review: bool, flags: list[Flag], link: Optional[LinkResult], rules: RuleSet) -> Route:
    if is_contradiction(flags, rules):
        return Route.CONTRADICTION
    unlinked = link is None or link.method == "unlinked"
    low_flags = any(f.severity not in rules.contradiction_severities for f in flags)
    if needs_review or unlinked or low_flags:
        return Route.REVIEW
    return Route.AUTO
