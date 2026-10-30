"""seam_holdback: the live-dictation hold-back holds exactly the words the
dictation map can still rewrite once the next utterance arrives.

The map matches its keys with sre's IGNORECASE, which folds i, I, İ and ı
together. A word ending the utterance that the map takes for a key's first
word but the hold-back does not is formatted and typed at once — and the
client never takes typed text back.
"""

from faster_whisper_backend.pipeline import dictation_map
from faster_whisper_backend.pipeline import seam_holdback as sh


def _held(raw, m):
    spec = sh.build_spec([dictation_map.compile_map(m)[2]])
    return raw[sh.held_start(raw, spec):].strip()


def test_a_turkish_dotted_or_dotless_i_is_held_like_the_map_matches_it():
    m = {"istanbul havalimanı": "IST"}
    pattern, replace, _ = dictation_map.compile_map(m)
    for word in ("istanbul", "İstanbul", "ıstanbul", "ISTANBUL"):
        assert pattern.sub(replace, f"nach {word} havalimanı") == "nach IST"
        assert _held(f"Wir fliegen nach {word}", m) == word


def test_a_key_written_with_a_turkish_i_holds_the_plain_i_text():
    m = {"İzmir yolu": "IZY", "İki nokta": ":"}
    pattern, replace, _ = dictation_map.compile_map(m)
    for word in ("izmir", "IZMIR", "İzmir"):
        assert pattern.sub(replace, f"nach {word} yolu") == "nach IZY"
        assert _held(f"nach {word}", m) == word
    assert pattern.sub(replace, "Tamam. iki nokta") == "Tamam:"
    assert _held("Tamam. iki", m) == "iki"


def test_a_joiner_key_with_a_turkish_i_is_held():
    assert _held("Eisen BİNDESTRİCH", {"Bindestrich": "-"}) == "BİNDESTRİCH"
