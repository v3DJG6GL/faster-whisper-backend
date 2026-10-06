"""core.languages — the one English language-name table."""

from faster_whisper.tokenizer import _LANGUAGE_CODES

from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.core import languages


def test_whisper_table_matches_faster_whisper():
    assert set(languages.WHISPER_LANGUAGE_NAMES) == set(_LANGUAGE_CODES)
    assert settings_schema.WHISPER_LANGUAGE_CODES == set(_LANGUAGE_CODES)


def test_extra_codes_are_not_whisper_codes():
    assert not set(languages.EXTRA_LANGUAGE_NAMES) & set(_LANGUAGE_CODES)


def test_language_name_lookup_order():
    assert languages.language_name("de") == "German"
    assert languages.language_name("my") == "Burmese"
    assert languages.language_name("zh-Hant") == "Traditional Chinese"
    assert languages.language_name("ZH-HANT") == "Traditional Chinese"
    assert languages.language_name("zh-TW") == "Chinese"       # region → base
    assert languages.language_name("pt-BR") == "Portuguese"
    assert languages.language_name("nb") == "Norwegian Bokmål"
    assert languages.language_name("rm") == "Rm"               # title-case fallback
    assert languages.language_name(None) == ""
    assert languages.language_name("") == ""


def test_language_label_keeps_the_subtag_in_the_title():
    label = languages.language_label
    assert label("de") == "German" and label("pt-BR") == "Portuguese (BR)"
    assert label("zh-TW") == "Chinese (TW)" and label("rm-CH") == "Rm (CH)"
    # A code the table names in full keeps that name.
    assert label("zh-Hant") == "Traditional Chinese"
    assert label("") == "Unknown" and label(None) == "Unknown"


def test_lookup_full_code_then_base():
    table = {"zh-hant": "Hant", "zh": "Zh"}
    assert languages.lookup(table, "zh-Hant") == "Hant"
    assert languages.lookup(table, "ZH-TW") == "Zh"
    assert languages.lookup(table, "de") is None
    assert languages.lookup(table, None) is None


def test_canonical_code():
    c = languages.canonical_code
    assert c("deu") == "de" and c("EN") == "en" and c("jav") == "jw"
    assert c("zh_hant") == "zh-Hant" and c("sr-Latn") == "sr"
    assert c("pt-br") == "pt-BR" and c("haw") == "haw"
    assert c("rm") is None and c("multilingual") is None and c("") is None
    # Script ahead of a region; withdrawn and alternate 639-1 spellings.
    assert c("zh-Hant-TW") == "zh-Hant" and c("zh_Hant_HK") == "zh-Hant"
    assert c("sr-Latn-RS") == "sr"
    assert c("iw") == "he" and c("in") == "id" and c("jv") == "jw"
    # Every mapped code lands on a code the table names.
    assert all(v in languages.ALL_LANGUAGE_NAMES
               for v in languages._ISO639_3_TO_1.values())


def test_iso639_2t_terminology_codes():
    t = languages.iso639_2t
    assert t("en") == "eng" and t("de") == "deu" and t("fr") == "fra"
    assert t("zh") == "zho" and t("nl") == "nld" and t("pt-BR") == "por"
    assert t("jw") == "jav" and t("jv") == "jav"
    assert t("iw") == "heb" and t("in") == "ind"
    assert t("yue") == "yue" and t("haw") == "haw" and t("deu") == "deu"
    assert t("xx") == "und" and t("") == "und" and t(None) == "und"
    # The extra translation languages get their tag, not "und".
    assert t("nb") == "nob" and t("ug") == "uig" and t("zu") == "zul"
    # Every named language has a real tag.
    assert all(t(code) != "und" for code in languages.ALL_LANGUAGE_NAMES)


def test_same_language():
    s = languages.same_language
    assert s("pt-BR", "pt") and s("de", "DE") and s("de-CH", "de_AT")
    assert not s("zh-Hant", "zh") and not s("zh", "zh-Hans")
    assert s("zh-Hant", "zh-hant-TW") and not s("zh-Hant", "zh-Hans")
    assert not s("en", "de") and not s("", "en") and not s("en", None)
    # Withdrawn / standard / 639-3 spellings name the same language.
    assert s("iw", "he") and s("jv", "jw") and s("deu", "de") and s("in", "id")
    assert not s("iw", "en")


def test_language_codes_normalises_case_and_dedupes():
    codes = languages.language_codes
    assert codes("DE,EN,fr-CA,fr-ca") == ["de", "en", "fr-CA"]
    assert codes("zh-hant, pt_br ,,x,toolong-") == ["zh-Hant", "pt-BR"]
    assert codes("es-419,es-419") == ["es-419"]
    assert codes(None) == [] and codes("") == []


def test_language_codes_limit_bounds_a_huge_csv():
    """A 1 MiB translate_to form field must not cost an O(n^2) walk."""
    import time

    csv = ",".join(f"aa-{i:04d}" for i in range(10_000)) * 5
    t0 = time.perf_counter()
    got = languages.language_codes(csv, limit=11)
    assert time.perf_counter() - t0 < 0.05
    assert len(got) == 11
    # Unbounded, the dedup is still linear (a set, not a list scan).
    t0 = time.perf_counter()
    assert len(languages.language_codes(csv)) == 10_000
    assert time.perf_counter() - t0 < 2.0
