"""Phase 4 checks: one passing and one failing case per rule R1-R10, the rule loader, the linker."""
import copy

import pytest

from lre.config import load_config
from lre.link.linker import build_index, link
from lre.schema import Holding, Owner, Plot, RegisterRow, Remarks
from lre.validate.engine import RuleError, compile_check, is_contradiction, load_rules, validate


@pytest.fixture(scope="module")
def rules():
    return load_rules("rules.yaml")


def _holding(**kw) -> Holding:
    h = Holding(
        holding_id="V000-P0001-R01", village_code="V000", khewat_no="41", khatauni_no="57",
        owners=[Owner("राम सिंह", "s/o", "हरी सिंह", "1/2"), Owner("मोहन लाल", "s/o", "हरी सिंह", "1/2")],
        cultivator="khudkasht",
        plots=[Plot("112//3", 132, "chahi"), Plot("112//4", 88, "barani")],
        total_area_marla=220,
        remarks=Remarks(raw="इंतकाल नं. 2231", mutation_no="2231"),
    )
    for k, v in kw.items():
        setattr(h, k, v)
    return h


def _row(h: Holding) -> RegisterRow:
    return RegisterRow(register_row_id="V000-41", village_code="V000", khewat_no=h.khewat_no,
                       khatauni_no=h.khatauni_no, owners=copy.deepcopy(h.owners), cultivator=h.cultivator,
                       plots=copy.deepcopy(h.plots), mutation_history=["1500", "2231"], last_mutation_ref="2231")


def _fired(h, row, rules, index=None):
    return {f.rule for f in validate(h, row, rules, index)}


def test_consistent_holding_fires_nothing(rules):
    h = _holding()
    assert _fired(h, _row(h), rules) == set()


def test_r1_shares_sum(rules):
    h = _holding()
    h.owners[0].share = "2/3"
    assert "R1" in _fired(h, None, rules)
    h.owners[0].share = ""          # unreadable share
    assert "R1" in _fired(h, None, rules)


def test_r2_written_total(rules):
    assert "R2" in _fired(_holding(total_area_marla=230), None, rules)
    assert "R2" not in _fired(_holding(total_area_marla=None), None, rules)


def test_r3_units(rules):
    h = _holding()
    h.plots[0].area_raw = "6-23"
    assert "R3" in _fired(h, None, rules)
    h.plots[0].area_raw = "6-12"
    assert "R3" not in _fired(h, None, rules)


def test_r4_vocabulary(rules):
    h = _holding()
    h.plots[1].land_type = None
    flags = validate(h, None, rules)
    assert any(f.rule == "R4" for f in flags)
    assert not is_contradiction(flags, rules)       # low severity: review, not contradiction


def test_r5_owner_set(rules):
    h = _holding()
    row = _row(h)
    row.owners[1].name = "कमला रानी"
    assert "R5" in _fired(h, row, rules)
    row.owners[1].name = "मोहन लाल"
    assert "R5" not in _fired(h, row, rules)


def test_r5_tolerates_spelling_variants(rules):
    h = _holding()
    row = _row(h)
    row.owners[0].name = "राम सिंघ"            # benign variant
    row.owners[0].relation = "पुत्र"
    assert _fired(h, row, rules) == set()


def test_r6_share(rules):
    h = _holding()
    row = _row(h)
    row.owners[0].share, row.owners[1].share = "1/3", "2/3"
    assert "R6" in _fired(h, row, rules)


def test_r7_area_with_tolerance(rules):
    h = _holding()
    row = _row(h)
    row.plots[0].area_marla = 133               # 1 marla: within max(1, 2 %)
    assert "R7" not in _fired(h, row, rules)
    row.plots[0].area_marla = 150
    assert "R7" in _fired(h, row, rules)


def test_r7_exact_profile_has_no_tolerance():
    exact = load_rules("rules.yaml", profile="exact")
    h = _holding()
    row = _row(h)
    row.plots[0].area_marla = 133
    assert "R7" in _fired(h, row, exact)


def test_r8_khasra_moved(rules):
    h = _holding()
    row = _row(h)
    row.plots = row.plots[:1]                    # 112//4 now belongs to khewat 42
    other = RegisterRow("V000-42", "V000", "42", owners=[Owner("x", share="1/1")], plots=[Plot("112//4", 88)])
    index = build_index([row, other]).khasra_owner
    assert "R8" in _fired(h, row, rules, index)
    assert "R8" not in _fired(h, _row(h), rules, build_index([_row(h)]).khasra_owner)


def test_r9_cultivator(rules):
    h = _holding()
    row = _row(h)
    row.cultivator = "सोहन लाल s/o बचन सिंह"
    assert "R9" in _fired(h, row, rules)


def test_r10_mutation(rules):
    h = _holding()
    row = _row(h)
    row.mutation_history = ["1500"]
    assert "R10" in _fired(h, row, rules)
    assert "R10" not in _fired(h, _row(h), rules)


def test_digits_and_relations_are_normalised(rules):
    h = _holding()
    row = _row(h)
    row.plots[0].khasra = "११२//३"
    row.owners[1].relation = "वल्द"
    assert _fired(h, row, rules) == set()


def test_flags_carry_evidence(rules):
    h = _holding()
    row = _row(h)
    row.owners[0].share, row.owners[1].share = "1/3", "2/3"
    f = next(f for f in validate(h, row, rules) if f.rule == "R6")
    assert f.evidence["scan_share"] in ("1/2",) and f.evidence["register_share"] in ("1/3", "2/3")


@pytest.mark.parametrize("expr", ["__import__('os')", "fact.__class__", "open('x')", "x + 1", "[c for c in 'ab']"])
def test_unsafe_checks_rejected(expr):
    with pytest.raises(RuleError):
        compile_check(expr, "T")


def test_unknown_profile_rejected():
    with pytest.raises(RuleError):
        load_rules("rules.yaml", profile="nope")


def test_linker_khewat_and_khasra_fallback():
    cfg = load_config().linking
    h = _holding()
    row = _row(h)
    assert link(h, build_index([row]), cfg).method == "khewat"
    row.khewat_no, row.register_row_id = "141", "V000-141"   # renumbered between registers
    res = link(h, build_index([row]), cfg)
    assert (res.method, res.register_row_id) == ("khasra_jaccard", "V000-141")
    h2 = _holding(plots=[Plot("999//1", 10, "chahi")], khewat_no="77")
    assert link(h2, build_index([row]), cfg).method == "unlinked"
