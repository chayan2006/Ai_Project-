"""Phase 0 checks: config, data contracts, register I/O, logging, checkpoints."""
import logging

import pytest

from lre import pipeline
from lre.config import ConfigError, load_config
from lre.io import RegisterSchemaError, load_register, read_json, save_register, write_json
from lre.logging_utils import MaskNamesFilter, mask
from lre.schema import (Flag, Holding, LinkResult, Owner, Plot, RegisterRow, Remarks, Route,
                        holding_from_dict, to_dict)


def test_default_config_matches_yaml():
    cfg = load_config("config.yaml")
    assert cfg.preprocess.sauvola_window == 31 and cfg.preprocess.sauvola_k == 0.2
    assert cfg.routing.alpha == 0.98
    assert cfg.synth.fonts.test and set(cfg.synth.fonts.test).isdisjoint(cfg.synth.fonts.train)
    assert len(cfg.hash()) == 12


def test_unknown_config_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("preprocess:\n  sauvola_windw: 31\n")
    with pytest.raises(ConfigError, match="sauvola_windw"):
        load_config(p)


def _holding():
    return Holding(
        holding_id="V000-P0007-R01", village_code="V000", khewat_no="41", khatauni_no="57",
        owners=[Owner("राम सिंह", "s/o", "हरी सिंह", "1/2", 0.96), Owner("मोहन लाल", "s/o", "हरी सिंह", "1/2")],
        plots=[Plot("112//3", 132, "chahi", 0.97)],
        remarks=Remarks(raw="इंतकाल नं. 2231", mutation_no="2231", date="2019-03-12"),
        link=LinkResult("V000-41", "khewat", 1.0),
        flags=[Flag("R6", "share_differs", "cross", "high", evidence={"scan": "1/2", "register": "1/3"})],
        route=Route.CONTRADICTION,
    )


def test_holding_json_round_trip(tmp_path):
    h = _holding()
    write_json(tmp_path / "h.json", h)
    back = holding_from_dict(read_json(tmp_path / "h.json"))
    assert back == h
    assert to_dict(h)["route"] == "CONTRADICTION"
    assert sum(o.share_fraction for o in h.owners) == 1


def test_register_round_trip_and_schema_error(tmp_path):
    rows = [RegisterRow("V000-41", "V000", "41", "57", owners=[Owner("राम सिंह", "s/o", "हरी सिंह", "1/1")],
                        plots=[Plot("112//3", 132, "chahi")], mutation_history=["2231"], last_mutation_ref="2231")]
    save_register(rows, tmp_path / "r.csv")
    assert load_register(tmp_path / "r.csv") == rows
    assert load_register(tmp_path / "r.csv", village="V999") == []
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    with pytest.raises(RegisterSchemaError, match="missing columns"):
        load_register(tmp_path / "bad.csv")


def test_names_masked_in_logs():
    assert mask("owner राम सिंह share 1/2") == "owner *** *** share 1/2"
    rec = logging.LogRecord("lre", logging.INFO, "", 0, "linked %s", ("राम सिंह",), None)
    MaskNamesFilter().filter(rec)
    assert "राम" not in rec.getMessage()


def test_checkpoint_resume(tmp_path):
    pipeline.save_checkpoint(tmp_path, "P1", "ingest", {"ok": 1})
    pipeline.save_checkpoint(tmp_path, "P1", "preprocess", {"ok": 1})
    assert pipeline.last_completed_stage(tmp_path, "P1") == "preprocess"
    assert pipeline.load_checkpoint(tmp_path, "P1", "layout") is None
