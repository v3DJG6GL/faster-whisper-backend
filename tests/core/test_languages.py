"""core.languages — the one English language-name table."""

from faster_whisper.tokenizer import _LANGUAGE_CODES

from faster_whisper_backend import config_store
from faster_whisper_backend.core import languages


def test_whisper_table_matches_faster_whisper():
    assert set(languages.WHISPER_LANGUAGE_NAMES) == set(_LANGUAGE_CODES)
    assert config_store.WHISPER_LANGUAGE_CODES == set(_LANGUAGE_CODES)


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


def test_canonical_code():
    c = languages.canonical_code
    assert c("deu") == "de" and c("EN") == "en" and c("jav") == "jw"
    assert c("zh_hant") == "zh-Hant" and c("sr-Latn") == "sr"
    assert c("pt-br") == "pt-BR" and c("haw") == "haw"
    assert c("rm") is None and c("multilingual") is None and c("") is None
    # Every mapped code lands on a code the table names.
    assert all(v in languages.ALL_LANGUAGE_NAMES
               for v in languages._ISO639_3_TO_1.values())
