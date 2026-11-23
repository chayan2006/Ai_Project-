"""Contradiction metrics against SynJama labels (report Table 10; Phase 4 check in D.3)."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..config import Config
from ..io import load_register
from ..schema import Holding, holding_from_dict
from ..validate.engine import RuleSet, is_contradiction
from ..validate.runner import validate_holdings, write_contradiction_report

POSITIVE_KINDS = {"contradiction", "intra"}


def load_labels(path: str | Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["holding_id"]:
                out[row["holding_id"]].append(row)
    return out


def truth_from_labels(labels: dict[str, list[dict]], holding_ids: list[str]) -> dict[str, bool]:
    return {hid: any(lab["kind"] in POSITIVE_KINDS for lab in labels.get(hid, [])) for hid in holding_ids}


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else None
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def load_gt_holdings(data_dir: str | Path) -> list[Holding]:
    with open(Path(data_dir) / "holdings_gt.jsonl", encoding="utf-8") as fh:
        return [holding_from_dict(json.loads(line)) for line in fh]


def evaluate_validation_on_gt(data_dir: str | Path, cfg: Config, rules: RuleSet) -> dict:
    """Run linking + rules on ground-truth fields. The Phase 4 check needs precision 1.0 and recall 1.0."""
    data_dir = Path(data_dir)
    holdings = load_gt_holdings(data_dir)
    rows = load_register(data_dir / "register.csv")
    labels = load_labels(data_dir / "labels.csv")
    validate_holdings(holdings, rows, rules, cfg.linking)

    truth = truth_from_labels(labels, [h.holding_id for h in holdings])
    tp = fp = fn = 0
    fired_by_type: dict[str, Counter] = defaultdict(Counter)
    expected_hit: dict[str, list[bool]] = defaultdict(list)
    false_pos, missed = [], []
    link_ok, link_methods = [], Counter()
    for h in holdings:
        flagged = is_contradiction(h.flags, rules)
        fired = {f.rule for f in h.flags}
        labs = labels.get(h.holding_id, [])
        tp += flagged and truth[h.holding_id]
        fp += flagged and not truth[h.holding_id]
        fn += (not flagged) and truth[h.holding_id]
        if flagged and not truth[h.holding_id]:
            false_pos.append({"holding_id": h.holding_id, "rules": sorted(fired),
                              "labels": [lab["type"] for lab in labs],
                              "evidence": [f.evidence for f in h.flags][:3]})
        if truth[h.holding_id] and not flagged:
            missed.append({"holding_id": h.holding_id, "labels": [lab["type"] for lab in labs]})
        for lab in labs:
            fired_by_type[f"{lab['kind']}:{lab['type']}"].update(fired or {"-"})
            if lab["rules"]:
                expected_hit[lab["type"]].append(set(lab["rules"].split(";")) <= fired)
        renum = next((lab for lab in labs if lab["kind"] == "renumber"), None)
        expected_row = f"{h.village_code}-{renum['register_value'] if renum else h.khewat_no}"
        link_ok.append(h.link is not None and h.link.register_row_id == expected_row)
        link_methods[h.link.method if h.link else "none"] += 1

    m = {
        "holdings": len(holdings),
        "positives": sum(truth.values()),
        "profile": rules.profile,
        "holding_level": prf(tp, fp, fn),
        "link_accuracy": sum(link_ok) / max(1, len(link_ok)),
        "link_methods": dict(link_methods),
        "expected_rule_fired": {t: sum(v) / len(v) for t, v in sorted(expected_hit.items())},
        "rules_fired_by_label": {t: dict(c) for t, c in sorted(fired_by_type.items())},
        "false_positives": false_pos[:20],
        "missed": missed[:20],
    }
    m["acceptance_passed"] = m["holding_level"]["precision"] == 1.0 and m["holding_level"]["recall"] == 1.0
    report = Path(cfg.paths.results_dir) / f"contradiction_report_gt_{data_dir.name}.csv"
    m["flagged_holdings"] = write_contradiction_report(holdings, rules, report)
    m["report"] = str(report)
    return m
