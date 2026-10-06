"""English language names: the one table every language label comes from.

``WHISPER_LANGUAGE_NAMES`` holds Whisper's 100 codes (the keys of
``faster_whisper.tokenizer._LANGUAGE_CODES``, pinned by a test) — the admin
language picker, the config validators and the translation prompts all read
it. ``EXTRA_LANGUAGE_NAMES`` adds the codes translation models support that
Whisper cannot transcribe (``translation._FAMILIES`` language lists).
"""

import re

WHISPER_LANGUAGE_NAMES: "dict[str, str]" = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese",
    "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian",
    "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan", "br": "Breton",
    "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh",
    "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian",
    "fi": "Finnish", "fo": "Faroese", "fr": "French", "gl": "Galician",
    "gu": "Gujarati", "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole",
    "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian",
    "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jw": "Javanese",
    "ka": "Georgian", "kk": "Kazakh", "km": "Khmer", "kn": "Kannada",
    "ko": "Korean", "la": "Latin", "lb": "Luxembourgish", "ln": "Lingala",
    "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy",
    "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian",
    "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese",
    "ne": "Nepali", "nl": "Dutch", "nn": "Nynorsk", "no": "Norwegian",
    "oc": "Occitan", "pa": "Punjabi", "pl": "Polish", "ps": "Pashto",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit",
    "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian",
    "sn": "Shona", "so": "Somali", "sq": "Albanian", "sr": "Serbian",
    "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "tg": "Tajik", "th": "Thai", "tk": "Turkmen",
    "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar", "uk": "Ukrainian",
    "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese", "yi": "Yiddish",
    "yo": "Yoruba", "zh": "Chinese", "yue": "Cantonese",
}

EXTRA_LANGUAGE_NAMES: "dict[str, str]" = {
    "zh-Hant": "Traditional Chinese", "ug": "Uyghur",
    "nb": "Norwegian Bokmål", "zu": "Zulu",
}

ALL_LANGUAGE_NAMES: "dict[str, str]" = {
    **WHISPER_LANGUAGE_NAMES, **EXTRA_LANGUAGE_NAMES}

# Lowercase lookup ("zh-hant" finds "zh-Hant").
_NAMES = {code.lower(): name for code, name in ALL_LANGUAGE_NAMES.items()}


def lookup(table: "dict[str, str]", code: "str | None") -> "str | None":
    """A lowercase-keyed table's entry for a language code: the full code
    first ("zh-hant"), then its base ("pt-BR" → "pt"); None when neither."""
    low = (code or "").strip().lower()
    return table.get(low) or table.get(low.split("-")[0])


def language_name(code: "str | None") -> str:
    """English name for a language code: the full code first ("zh-Hant"),
    then its base ("pt-BR" → "Portuguese"); unknown codes are title-cased
    ("rm" → "Rm")."""
    if not code:
        return ""
    return lookup(_NAMES, code) or code.strip().lower().split("-")[0].title()


def language_label(code: "str | None") -> str:
    """A track title for a language code: the name the table gives the full
    code ("zh-Hant" → "Traditional Chinese"), else the base name with the
    subtag kept ("pt-BR" → "Portuguese (BR)"); "Unknown" for no code."""
    raw = (code or "").strip()
    if not raw:
        return "Unknown"
    base, _, sub = raw.partition("-")
    if not sub or raw.lower() in _NAMES:
        return language_name(raw)
    return f"{language_name(base)} ({sub.upper()})"


# ISO 639-3 → the table's 639-1 code, for the codes the table names: model
# cards list "deu"/"eng_Latn" as often as "de". The individual codes behind a
# macrolanguage ("cmn", "arb", "zsm"…) are the ones NLLB-style cards use.
_ISO639_3_TO_1: "dict[str, str]" = dict(pair.split("=")[::-1] for pair in """
    af=afr am=amh ar=ara as=asm az=aze ba=bak be=bel bg=bul bn=ben bo=bod
    br=bre bs=bos ca=cat cs=ces cy=cym da=dan de=deu el=ell en=eng es=spa
    et=est eu=eus fa=fas fi=fin fo=fao fr=fra gl=glg gu=guj ha=hau he=heb
    hi=hin hr=hrv ht=hat hu=hun hy=hye id=ind is=isl it=ita ja=jpn jw=jav
    ka=kat kk=kaz km=khm kn=kan ko=kor la=lat lb=ltz ln=lin lo=lao lt=lit
    lv=lav mg=mlg mi=mri mk=mkd ml=mal mn=mon mr=mar ms=msa mt=mlt my=mya
    ne=nep nl=nld nn=nno no=nor nb=nob oc=oci pa=pan pl=pol ps=pus pt=por
    ro=ron ru=rus sa=san sd=snd si=sin sk=slk sl=slv sn=sna so=som sq=sqi
    sr=srp su=sun sv=swe sw=swa ta=tam te=tel tg=tgk th=tha tk=tuk tl=tgl
    tr=tur tt=tat uk=ukr ur=urd uz=uzb vi=vie yi=yid yo=yor zh=zho ug=uig
    zu=zul zh=cmn ar=arb fa=pes ms=zsm et=ekk lv=lvs mn=khk uz=uzn sw=swh
    mg=plt ne=npi az=azj ps=pbt sq=als yi=ydd
""".split())
# 639-1 spellings the table does not use: the withdrawn codes YouTube and
# yt-dlp still send ("iw" Hebrew, "in" Indonesian, "ji" Yiddish, "mo"
# Moldavian) and the standard "jv" for Whisper's "jw".
_LEGACY_639_1: "dict[str, str]" = {
    "iw": "he", "in": "id", "ji": "yi", "mo": "ro", "jv": "jw"}

# The inverse: 639-1 → ISO 639-2/T (terminology codes — `deu` not `ger`,
# `fra` not `fre`: what ffmpeg writes and what Matroska/MP4 players expect).
# Built in reverse so the FIRST 3-letter code per language wins ("zh" →
# "zho", not "cmn").
_ISO639_1_TO_2T: "dict[str, str]" = {
    one: three for three, one in reversed(_ISO639_3_TO_1.items())}


def iso639_2t(code: "str | None") -> str:
    """The 639-2/T tag for a client language code ("pt-BR" → "por"); the
    region is dropped (containers store 639-2 only), an ISO 639-3
    individual code the table maps is folded to its macrolanguage ("cmn" →
    "zho", "arb" → "ara": players do not know the 639-3 spelling), a 3-letter
    code the table lacks already is one ("yue", "haw"), and an unknown code
    becomes "und" rather than an invalid tag."""
    base = (code or "").strip().lower().split("-")[0]
    base = _LEGACY_639_1.get(base, _ISO639_3_TO_1.get(base, base))
    if len(base) == 3 and base.isalpha() and base not in _ISO639_1_TO_2T:
        return base
    return _ISO639_1_TO_2T.get(base, "und")


def canonical_code(code: str) -> "str | None":
    """A model card's language code as this table spells it, or None when
    the table cannot name it ("rm", "multilingual"). An ISO 639-3 base maps
    to its 639-1 code ("deu" → "de"); a script subtag stays only where the
    table names it ("zh_hant" → "zh-Hant", "eng_Latn" → "en"); a region
    stays ("pt-br" → "pt-BR"); a script ahead of a region keeps the script
    rule ("zh_Hant_TW" → "zh-Hant"); a withdrawn 639-1 code maps to its
    successor ("iw" → "he")."""
    base, _, sub = code.strip().replace("_", "-").partition("-")
    base = base.lower()
    base = _LEGACY_639_1.get(base, _ISO639_3_TO_1.get(base, base))
    if base not in _NAMES:
        return None
    script = sub.split("-", 1)[0]
    if len(script) == 4 and script.isalpha():
        full = f"{base}-{script.title()}"
        return full if full.lower() in _NAMES else base
    full = f"{base}-{sub.upper()}" if sub else base
    return full if TRANSLATE_CODE_RE.match(full) else None


def same_language(a: "str | None", b: "str | None") -> bool:
    """Whether two codes name the same written language: the base subtag
    decides ("pt-BR" == "pt"), except that a script subtag on either side
    must match ("zh" != "zh-Hant": Whisper's Chinese is not Traditional
    Chinese). False when either is empty."""
    if not a or not b:
        return False
    a_base, _, a_sub = a.strip().lower().replace("_", "-").partition("-")
    b_base, _, b_sub = b.strip().lower().replace("_", "-").partition("-")
    # Through the same spelling tables canonical_code uses: Whisper says
    # "he"/"jw", a client may send the withdrawn "iw" or the standard "jv".
    a_base = _LEGACY_639_1.get(a_base, _ISO639_3_TO_1.get(a_base, a_base))
    b_base = _LEGACY_639_1.get(b_base, _ISO639_3_TO_1.get(b_base, b_base))
    if a_base != b_base:
        return False

    def script(sub: str) -> str:
        head = sub.split("-", 1)[0]
        return head if len(head) == 4 and head.isalpha() else ""
    return script(a_sub) == script(b_sub)


# One translation language code: a 2-3 letter base ("en", "de", "gsw") plus an
# optional BCP-47-ish subtag ("fr-CA", "zh-Hant").
TRANSLATE_CODE_RE = re.compile(r"\A[a-z]{2,3}(-[A-Za-z0-9]{2,8})?\Z")


def _normalise_code(code: str) -> str:
    """One csv entry in the table's spelling: "_" → "-", a lowercase base,
    a script subtag title-cased ("zh-hant" → "zh-Hant"), a 2-letter region
    upper-cased ("fr-ca" → "fr-CA"); any other subtag stays as given."""
    base, sep, sub = code.strip().replace("_", "-").partition("-")
    base = base.lower()
    if not sep:
        return base
    if len(sub) == 4 and sub.isalpha():
        sub = sub.title()
    elif len(sub) == 2 and sub.isalpha():
        sub = sub.upper()
    return f"{base}-{sub}"


def language_codes(csv: "str | None", limit: "int | None" = None) -> "list[str]":
    """A csv of codes ("en,fr-CA") → deduped ordered list of the well-formed
    ones, normalised so "DE" is "de" and "fr-ca" the same target as "fr-CA";
    malformed entries drop silently. `limit` stops the walk once that many
    codes are collected — a per-request csv can carry a 1 MiB form field."""
    out: "list[str]" = []
    seen: "set[str]" = set()
    for code in (csv or "").split(","):
        if limit is not None and len(out) >= limit:
            break
        code = _normalise_code(code)
        if code and code.lower() not in seen and TRANSLATE_CODE_RE.match(code):
            seen.add(code.lower())
            out.append(code)
    return out
