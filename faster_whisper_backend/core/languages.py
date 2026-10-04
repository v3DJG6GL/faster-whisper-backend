"""English language names: the one table every language label comes from.

``WHISPER_LANGUAGE_NAMES`` holds Whisper's 100 codes (the keys of
``faster_whisper.tokenizer._LANGUAGE_CODES``, pinned by a test) — the admin
language picker, the config validators and the translation prompts all read
it. ``EXTRA_LANGUAGE_NAMES`` adds the codes translation models support that
Whisper cannot transcribe (``translation._FAMILIES`` language lists).
"""

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

# Lowercase lookup over both tables ("zh-hant" finds "zh-Hant").
_NAMES = {code.lower(): name for code, name in
          {**WHISPER_LANGUAGE_NAMES, **EXTRA_LANGUAGE_NAMES}.items()}


def language_name(code: "str | None") -> str:
    """English name for a language code: the full code first ("zh-Hant"),
    then its base ("pt-BR" → "Portuguese"); unknown codes are title-cased
    ("rm" → "Rm")."""
    if not code:
        return ""
    low = code.strip().lower()
    base = low.split("-")[0]
    return _NAMES.get(low) or _NAMES.get(base) or base.title()
