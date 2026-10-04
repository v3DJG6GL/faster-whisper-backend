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


def language_name(code: "str | None") -> str:
    """English name for a language code: the full code first ("zh-Hant"),
    then its base ("pt-BR" → "Portuguese"); unknown codes are title-cased
    ("rm" → "Rm")."""
    if not code:
        return ""
    low = code.strip().lower()
    base = low.split("-")[0]
    return _NAMES.get(low) or _NAMES.get(base) or base.title()


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


def canonical_code(code: str) -> "str | None":
    """A model card's language code as this table spells it, or None when
    the table cannot name it ("rm", "multilingual"). An ISO 639-3 base maps
    to its 639-1 code ("deu" → "de"); a script subtag stays only where the
    table names it ("zh_hant" → "zh-Hant", "eng_Latn" → "en"); a region
    stays ("pt-br" → "pt-BR")."""
    base, _, sub = code.strip().replace("_", "-").partition("-")
    base = _ISO639_3_TO_1.get(base.lower(), base.lower())
    if base not in _NAMES:
        return None
    if len(sub) == 4 and sub.isalpha():
        full = f"{base}-{sub.title()}"
        return full if full.lower() in _NAMES else base
    full = f"{base}-{sub.upper()}" if sub else base
    return full if TRANSLATE_CODE_RE.match(full) else None


# One translation language code: a 2-3 letter base ("en", "de", "gsw") plus an
# optional BCP-47-ish subtag ("fr-CA", "zh-Hant").
TRANSLATE_CODE_RE = re.compile(r"\A[a-z]{2,3}(-[A-Za-z0-9]{2,8})?\Z")


def language_codes(csv: "str | None") -> "list[str]":
    """A csv of codes ("en,fr-CA") → deduped ordered list of the well-formed
    ones; malformed entries drop silently."""
    out: "list[str]" = []
    for code in (csv or "").split(","):
        code = code.strip()
        if code and code not in out and TRANSLATE_CODE_RE.match(code):
            out.append(code)
    return out
