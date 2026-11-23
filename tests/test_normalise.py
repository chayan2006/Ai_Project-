from fractions import Fraction

import pytest

from lre.extract import normalise as N


def test_digits_round_trip():
    assert N.to_ascii_digits("११२//३") == "112//3"
    assert N.to_dev_digits("59-60") == "५९-६०"
    assert N.to_ascii_digits(N.to_dev_digits("0123456789")) == "0123456789"


@pytest.mark.parametrize("text,expected", [
    ("हिस्सा 1/2", Fraction(1, 2)),
    ("हिस्सा १/३", Fraction(1, 3)),
    ("2/4", Fraction(1, 2)),
    ("1/1", Fraction(1)),
    ("हिस्सा 1", Fraction(1)),
    ("1/0", None),
    ("", None),
])
def test_parse_share(text, expected):
    assert N.parse_share(text) == expected


def test_parse_area_and_units():
    a = N.parse_area("6-12 चाही")
    assert a == N.Area(6, 12) and a.total_marla == 132 and a.valid
    assert N.parse_area("०-१५ बारानी").total_marla == 15
    bad = N.parse_area("6-23")
    assert bad is not None and not bad.valid          # R3 must be able to see it
    assert N.parse_area("no area") is None
    assert N.marla_to_kanal_marla(132) == "6-12"


@pytest.mark.parametrize("text,expected", [
    ("बंजर कदीम", "banjar_kadim"), ("बंजर", "banjar"), ("12-04 गैर मुमकिन", "gair_mumkin"),
    ("chahi", "chahi"), ("नहरी", "nehri"), ("xyz", None),
])
def test_land_type(text, expected):
    assert N.normalise_land_type(text) == expected


def test_khasra():
    assert N.parse_khasra("११२ // ३") == "112//3"
    assert N.parse_khasra("119/1") == "119/1"
    assert N.parse_khasra("118") == "118"
    assert N.parse_khasra_list("112//3\n112//4, 118") == ["112//3", "112//4", "118"]
    assert N.is_valid_khasra("112//3") and not N.is_valid_khasra("112///3")


def test_khatauni_and_khewat():
    assert N.parse_khatauni("59-60") == ["59", "60"]
    assert N.parse_khatauni("५९, ६१") == ["59", "61"]
    assert N.normalise_khewat("०४१") == "41"
    assert N.normalise_khewat("41 मिन") == "41min"


def test_mutation():
    assert N.parse_mutation("इंतकाल नं. 2231 दि. 12-03-2019") == ("2231", "2019-03-12")
    assert N.parse_mutation("इंतकाल नं. २२९० जेरे गौर") == ("2290", None)
    assert N.parse_mutation("Mut. 2231 (12-03-2019)") == ("2231", "2019-03-12")
    assert N.parse_mutation("") == (None, None)


def test_relation_and_cultivator():
    assert N.normalise_relation("पुत्र") == "s/o"
    assert N.normalise_relation("बेवा") == "w/o"
    assert N.normalise_cultivator("खुदकाश्त") == "khudkasht"
    assert N.normalise_cultivator("सोहन लाल पुत्र बचन सिंह (मुजारा)") == "सोहन लाल s/o बचन सिंह"


def test_parse_owner_line():
    o = N.parse_owner_line("राम सिंह पुत्र हरी सिंह हिस्सा 1/2")
    assert (o.name, o.relation, o.parent, o.share) == ("राम सिंह", "s/o", "हरी सिंह", "1/2")
    o = N.parse_owner_line("सीता देवी पत्नी राम किशन हिस्सा १/३")
    assert (o.relation, o.share) == ("w/o", "1/3")
