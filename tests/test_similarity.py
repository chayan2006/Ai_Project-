from lre.validate.similarity import name_similarity, phonetic_key


def test_identical_and_variants_score_high():
    assert name_similarity("राम सिंह", "राम सिंह") == 1.0
    assert name_similarity("राम सिंह", "राम सिंघ") >= 0.85
    assert name_similarity("सुरेंद्र कुमार", "सुरेन्द्र कुमार") >= 0.85
    assert name_similarity("गुरप्रीत कौर", "गुरप्रीत कोर") >= 0.85
    assert name_similarity("राजबीर सिंह", "राजवीर सिंह") >= 0.85


def test_different_people_score_low():
    assert name_similarity("राम सिंह", "सीता देवी") < 0.6
    assert name_similarity("बलवंत सिंह", "कमला रानी") < 0.6
    assert name_similarity("", "राम") == 0.0


def test_phonetic_key_is_latin():
    k = phonetic_key("राम सिंह")
    assert k.isascii() and "ram" in k
