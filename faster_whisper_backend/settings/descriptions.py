"""
Per-field descriptions for the admin settings schema (settings/schema.py).

Pure data: imported by schema.py (every AdminConfig field's Pydantic
description comes from here via its _F() helper) and read by admin/routes.py
as the fallback description source.
"""

# =============================================================================
# Single source of truth for field descriptions
# =============================================================================
# Surfaced everywhere a description is shown:
#   - Pydantic Field(description=…) — see _F() helper below
#   - /settings/state payload — admin/routes.py adds .description from the
#     Pydantic model_fields
#   - /settings admin WebUI — fieldRow() renders it as a <div class="help">
#     line under each editor
# Edit a string here, every consumer reflects it on next reload. Wording is
# cross-validated against upstream docs (faster-whisper, OpenAI Whisper,
# Silero VAD, CTranslate2, uvicorn, Python logging) where authoritative.
FIELD_DESCRIPTIONS: dict[str, str] = {
    # --- Models ---
    "DEFAULT_MODEL":
        "Model loaded when a request sends 'whisper-1' or omits 'model'. "
        "Accepts any faster-whisper short name or HF repo id.",
    "ALLOWED_MODELS":
        "Allowlist of model names clients may request. Empty set lets any "
        "well-formed name (short name or org/name repo id) pass — risks "
        "unknown multi-GB downloads.",
    "MAX_LOADED_MODELS":
        "Max models kept hot in VRAM (LRU evicts beyond this). large-v3 "
        "~1.5 GB fp16, turbo/distill ~600 MB.",
    "MODEL_IDLE_TIMEOUT_S":
        "Unload a model from VRAM/RAM after this many seconds without use. "
        "0 = disabled (default). Examples: 1800 = 30 min, 3600 = 1 h, "
        "14400 = 4 h. Background task wakes every 30 s to check.",
    "PRELOAD_MODELS":
        "Models eagerly loaded at startup so the first request skips the "
        "5-30 s warm-up. Empty = only DEFAULT_MODEL.",
    "MODEL_DEVICE":
        "Device to use for computation. cuda = NVIDIA GPU; cpu = host CPU. "
        "(faster-whisper)",
    "MODEL_COMPUTE_TYPE":
        "Numerical precision (CTranslate2).\n"
        "• float16: half-precision, weights + layers in FP16 "
        "(NVIDIA Volta+ / CUDA Compute Capability ≥ 7.0).\n"
        "• bfloat16: brain-float, half-precision "
        "(NVIDIA Ampere+ / CUDA Compute Capability ≥ 8.0).\n"
        "• int8: 8-bit weight quantization (smallest, fastest on CPU).\n"
        "• int8_float16: int8 weights + FP16 activations (smallest GPU footprint).\n"
        "• float32: full precision, largest + slowest.",
    "MODEL_DEVICE_FALLBACK":
        "Backup hardware target if the primary device fails to load "
        "(e.g. fall back to 'cpu' if CUDA is unavailable).",
    "MODEL_COMPUTE_TYPE_FALLBACK":
        "Backup precision used when the primary compute type isn't supported "
        "on the fallback device.",

    # --- Decode params (transcribe-time) ---
    "DEFAULT_LANGUAGE":
        "Language code such as 'en' or 'de'. If empty, the language is "
        "detected in the first 30 seconds of audio. (faster-whisper)",
    "DEFAULT_PROMPT":
        "Optional text passed as initial_prompt for the first window — "
        "useful for custom vocabularies or proper nouns to make those "
        "words more likely to be predicted. (OpenAI Whisper)",
    "BEAM_SIZE":
        "Beam size to use for decoding. Higher = better quality but slower. "
        "faster-whisper default 5.",
    "BEST_OF":
        "Number of independent sample trajectories when temperature > 0 "
        "(only on fallback-retry passes; the initial T=0 pass uses beam "
        "search instead). faster-whisper default 5.",
    "VAD_FILTER":
        "Enable voice activity detection (VAD) to filter out parts of the "
        "audio without speech. Uses the Silero VAD model. Reduces "
        "hallucinations in quiet audio.",
    "VAD_MIN_SILENCE_MS":
        "In the end of each speech chunk, wait this long before separating "
        "it. Default 500 ms (tuned to avoid splitting on short breaths).",
    "VAD_SPEECH_PAD_MS":
        "Final speech chunks are padded by this much on each side. "
        "Default 200 ms (prevents word-edge consonants from being clipped).",
    "VAD_THRESHOLD":
        "Speech threshold. Silero VAD outputs speech probabilities for "
        "each audio chunk; probabilities ABOVE this value are considered "
        "as SPEECH. Default 0.5; tune per dataset if needed.",
    "LEADING_SILENCE_PAD_MS":
        "Silence prepended to each uploaded file before decoding (file "
        "uploads only; timestamps are shifted back so the response matches "
        "the original audio). Prevents Whisper from dropping the opening "
        "words of a recording that starts mid-speech at t=0 — with "
        "DEFAULT_HOTWORDS set, the decoder can mistake a leading clause for "
        "text already covered by the hotword prompt and skip to the next "
        "sentence boundary. 0 disables. Default 500 ms.",
    "CONDITION_ON_PREVIOUS_TEXT":
        "If true, the previous output of the model is provided as a "
        "prompt for the next window. Disabling may make the text "
        "inconsistent across windows, but the model becomes less prone "
        "to getting stuck in failure loops (repetition; timestamps "
        "drifting out of sync). Default true.",
    "WORD_TIMESTAMPS_ENABLED":
        "Gate (not toggle): word timestamps run only when this is true AND "
        "the request asks (`timestamp_granularities[]=word` or "
        "`response_format=verbose_json`). False = force-off, even if asked.",
    "NO_SPEECH_THRESHOLD":
        "If the no_speech probability is higher than this value AND the "
        "average log-probability over sampled tokens is below "
        "LOG_PROB_THRESHOLD, the segment is dropped as silent. Default 0.6.",
    "LOG_PROB_THRESHOLD":
        "If the average log-probability over sampled tokens is below "
        "this value, treat the decode as failed (triggers a temperature-"
        "fallback retry). Default -1.0.",
    "COMPRESSION_RATIO_THRESHOLD":
        "If the gzip-compression ratio (zlib in the implementation) of "
        "the decoded text is above this value, treat the decode as "
        "failed (triggers a temperature-fallback retry). Catches "
        "repetition loops. Default 2.4.",
    "DEFAULT_HOTWORDS":
        "Persistent vocabulary biasing — re-injected into the prompt of "
        "every decoder window. Distinct from DEFAULT_PROMPT (which fades "
        "as decoded text accumulates): hotwords stay constant. Useful "
        "for domain terms, drug names, person names. Ignored when prefix "
        "is set per-call. (faster-whisper)",

    # --- Decode params (advanced) ---
    "TASK":
        "Whisper task when a request does not name one: 'transcribe' (source "
        "language) or 'translate' (into English — Whisper's only translation "
        "target). Clients override per request via the `task` form field or "
        "the /v1/audio/translations endpoint.",
    "TEMPERATURE":
        "Fallback ladder for decoding when compression / log-prob checks "
        "fail. Comma-separated floats (e.g. '0.0,0.2,0.4,0.6,0.8,1.0'). "
        "Lower / shorter ladders fail faster on distil models. (faster-whisper)",
    "PATIENCE":
        "Beam-search patience factor; >1 keeps the beam alive longer. "
        "Default 1.0. Try 1.5 if long sentences get clipped.",
    "LENGTH_PENALTY":
        "Beam-scoring length-norm exponent. >1 favors longer outputs, "
        "<1 favors shorter. Default 1.0. Tweak only if outputs are "
        "systematically too short or too padded.",
    "REPETITION_PENALTY":
        "Multiplies logit of already-emitted tokens by 1/penalty. >1 "
        "discourages loops. Default 1.0. Try 1.05–1.2 for stutter audio.",
    "NO_REPEAT_NGRAM_SIZE":
        "Hard ban on n-grams of this size repeating. 0 = off. Try 3 "
        "for stubborn repetition loops. Caveat: blocks legitimate "
        "repeats too.",
    "PROMPT_RESET_ON_TEMPERATURE":
        "When the temperature ladder fallback exceeds this value, drop "
        "the running text prompt to escape bad context. Default 0.5. "
        "Only relevant when CONDITION_ON_PREVIOUS_TEXT=True.",

    # --- Language detection (active when DEFAULT_LANGUAGE is empty) ---
    "MULTILINGUAL":
        "Re-run language detection on every segment instead of once. "
        "Default false. Enable for code-switching audio. Applies only when "
        "the language is auto-detected — a set language wins. Clients may "
        "override it per request ('multilingual'). (faster-whisper)",
    "LANGUAGE_DETECTION_THRESHOLD":
        "Min probability the top language token must reach for detection "
        "to be accepted. Default 0.5. Raise for stricter detection.",
    "LANGUAGE_DETECTION_SEGMENTS":
        "How many leading 30 s chunks to sample for language detection. "
        "Default 1. Bump to 2-5 if files start with silence/music.",

    # --- Anti-hallucination & token control ---
    "HALLUCINATION_SILENCE_THRESHOLD":
        "With WORD_TIMESTAMPS_ENABLED=true, skip silent stretches longer "
        "than this many seconds when a possible hallucination is detected. "
        "Default disabled; 0 = disabled too. Try 2.0 if Whisper invents "
        "'thanks for watching' filler in long silences.",
    "SEGMENT_MAX_WORDS_PER_S":
        "Drop a whole segment when its average speed is above this many "
        "words per second (number of words ÷ segment length). Catches "
        "segments that are made up from start to end: an echo of earlier "
        "text squeezed into a fraction of a second reaches 28–48 words per "
        "second, dictated speech stays under about 5. It cannot catch a "
        "made-up tail behind real words in the same segment, because the "
        "real words pull the average down — that is what "
        "SEGMENT_MAX_WORD_BURST_PER_S and SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS "
        "are for. Segments with fewer than 3 words are never dropped. "
        "Applies to batch and streaming finals. 0 = off. Default 10.",
    "SEGMENT_MAX_WORD_BURST_PER_S":
        "Cut the end of a segment when more than this many words start "
        "within its last second. The model sometimes does not stop after the "
        "last spoken word and keeps writing; those extra words have no audio "
        "behind them, so they pile up at the end of the recording (measured: "
        "12 word starts in one second, dictated speech at most 3). The text "
        "is cut at the first word of the pile; everything before it stays. "
        "Needs word timestamps. Applies to batch, streaming finals and live "
        "previews. 0 = off. Default 8.",
    "SEGMENT_ZERO_LENGTH_TAIL_MIN_WORDS":
        "Cut a made-up ending of at least this many words when the segment's "
        "last word has no duration (start = end). Words the model invents "
        "after the audio has ended get exactly zero length; spoken words do "
        "not. The ending is followed backwards over zero-length words, over "
        "words that are both very short (under 0.13 s) and very unsure "
        "(confidence under 0.15), and over the one unsure word that soaked up "
        "the leftover audio time. It also takes one confident word under a second long that sits "
        "between a zero-length word and at least two more zero-length words; "
        "otherwise it stops at the first confident word. A "
        "single zero-length word at the end is kept, so a real last word is "
        "never lost (which is why 1 is not accepted). Catches short made-up endings that are too few words "
        "for the burst check (measured: 'zu nehmen?, Fragezeichen' with "
        "confidence 0.01 / 0.10 behind a real word at 0.99). Needs word "
        "timestamps. Applies to batch, streaming finals and live previews. "
        "0 = off. Default 2.",
    "SEGMENT_REPEAT_COLLAPSE_MIN_REPEATS":
        "When a segment ends with the same phrase of 3 or more words "
        "repeated at least this many times in a row, keep the first copy and "
        "cut the rest. Safety net for repetition loops whose word timings "
        "look normal. Phrases of 1 or 2 words are never touched, so repeated "
        "commands (\"Neue Zeile Neue Zeile Neue Zeile\") stay. A repeat in "
        "the middle of a segment (a song refrain) is left alone; only a loop "
        "that runs to the end of the segment is cut. Works without word "
        "timestamps. Applies to batch, streaming finals and live previews. "
        "0 = off (1 is treated as off). Default 3.",
    "SEGMENT_HEAD_ECHO_MIN_WORDS":
        "Cut the start of a decode when it repeats the last words of its "
        "prompt — at least this many. Live dictation sends the previous "
        "sentences as the prompt, and the model sometimes writes their end "
        "again before the new speech, so it would be typed twice. Words the "
        "model repeats that way have no audio behind them: they are cut only "
        "when every one has no duration, is very short (under 0.13 s) and "
        "very unsure (confidence under 0.15), or is shorter than 0.07 s — one "
        "unsure word of any length may soak up the time before the real "
        "speech. A repeat you actually spoke has normal timings and stays. "
        "Only the first segment of a decode is checked; with no prompt (and "
        "no hotwords) there is nothing to compare. Needs word timestamps. "
        "Applies to batch, streaming finals and live previews. 0 = off; 1 is "
        "not accepted (a single repeated word is too common). Default 3.",
    "DECODE_SKIP_RESIDUAL_WINDOWS":
        "Stop decoding once a window reached the end of the audio (batch + "
        "streaming final). Whisper re-decodes the sub-second leftover after "
        "the last word as its own window — audio the previous window already "
        "saw and chose not to transcribe — and the temperature ladder can loop "
        "there for tens of seconds producing junk the confidence guard then "
        "drops. Full 30 s windows of long files are unaffected. The refused "
        "window is listed in the log block's Decode trace. Default on; turn "
        "off only to check whether a missing last word is caused by this rule.",
    "DECODE_TOKEN_CAP_PER_SECOND":
        "Limit every decode attempt to 30 + this many tokens per second of "
        "audio in the window (batch + streaming final). A decode that gets "
        "stuck repeating itself otherwise runs to the model's hard limit and "
        "costs 15-20 s before it is discarded. Real speech is about 3 tokens "
        "per second, so correct text is never touched; 30 s windows of long "
        "files are unaffected. An attempt that ran into the limit says "
        "'hit cap' in the log block's Decode trace. 0 = disabled. Default 10.",
    "SUPPRESS_BLANK":
        "Suppress blank token at start of decoder sampling. Default true. "
        "Almost never disable; only useful when debugging tokenizer behavior.",
    "SUPPRESS_TOKENS":
        "Comma-separated token IDs to ban from output. '-1' = expand to "
        "the model's default non-speech symbol set; '' = no suppression. "
        "Token IDs vary by tokenizer.",
    "SUPPRESS_CHARS":
        "Single chars to hard-mask during decoding. Each char is encoded "
        "via the loaded model's tokenizer (both bare and ' char' variants); "
        "single-token results are added to the effective suppress_tokens "
        "list — the decoder cannot emit them. Use '.,?!:;' for verbatim "
        "dictation: model can't auto-insert punctuation, so spoken 'Punkt' / "
        "'Komma' surface as words for the de-dictation-map PIPELINE_RULE to "
        "convert. Empty / unset = no extra suppression. Per-model overridable.",
    "PREPEND_PUNCTUATIONS":
        "With WORD_TIMESTAMPS_ENABLED, glue these characters onto the "
        "FOLLOWING word's timing. Locale-specific.",
    "APPEND_PUNCTUATIONS":
        "With WORD_TIMESTAMPS_ENABLED, glue these characters onto the "
        "PRECEDING word's timing. Add ؟ ، for Arabic, etc.",

    # --- Output wrappers ---
    "OUTPUT_PREFIX":
        "Plain text prepended to the final transcript text after the "
        "post-processing pipeline runs (before final whitespace trim). "
        "Empty / unset = no prefix. NOT a faster-whisper param.",
    "OUTPUT_SUFFIX":
        "Plain text appended to the final transcript text after the "
        "post-processing pipeline runs (before final whitespace trim). "
        "Empty / unset = no suffix. NOT a faster-whisper param.",

    # --- Load-time, hardware (advanced) ---
    "DOWNLOAD_ROOT":
        "Directory where HuggingFace model snapshots are cached. Empty = "
        "standard HF cache dir (~/.cache/huggingface).",
    "LOCAL_FILES_ONLY":
        "If true, never hit the network — only resolve from local cache. "
        "Default false. Use for air-gapped deploys.",
    "HF_TOKEN":
        "HuggingFace auth token for gated/private repos. Account-scoped. "
        "One token for everything Hugging Face: gated whisper repos and the "
        "pyannote diarization pipeline. Create one (read scope) at "
        "https://huggingface.co/settings/tokens — a fine-grained token also "
        "needs the public-gated-repos read permission.",
    "AUTO_CONVERT_HF_MODELS":
        "Auto-convert HuggingFace transformers Whisper models to CTranslate2 "
        "format on first load when no `model.bin` is present in the repo. "
        "Requires `pip install -r requirements-convert.txt` (transformers + "
        "torch + accelerate, ~2 GB). Output cached under CONVERTED_MODELS_DIR "
        "and loaded directly on subsequent starts. Conversion takes 1–3 min "
        "per model; happens during PRELOAD_MODELS startup or the first "
        "request that hits an unconverted model.",
    "CONVERT_QUANTIZATION":
        "On-disk weight quantisation when auto-converting HF→CT2. The "
        "saved precision is independent of MODEL_COMPUTE_TYPE (CT2 up- or "
        "down-casts at load). float16 = sweet spot for HF Whisper finetunes "
        "(matches source dtype, ~1.6 GB for large-v3-turbo). int8_float16 "
        "halves disk for marginal accuracy loss. Allowed: float32, float16, "
        "bfloat16, int16, int8, int8_float32, int8_float16, int8_bfloat16.",
    "CONVERTED_MODELS_DIR":
        "Output root for auto-converted CT2 models. Layout: "
        "<root>/<sanitized-id>/<quantization>/. Empty = ~/.cache/whisper-ct2.",
    "CPU_THREADS":
        "CPU threads for inference. 0 = library default (typically 4). "
        "Non-zero overrides OMP_NUM_THREADS for the worker pool.",
    "NUM_WORKERS":
        "Replicates the model so concurrent transcribe() calls run in true "
        "parallel. Default 1. Costs ~Nx VRAM for activation buffers.",
    "DEVICE_INDEX":
        "GPU index to bind to. Default 0. Set per-model on multi-GPU boxes "
        "to pin a model to a specific card.",

    # --- Preload & warm cache (advanced) ---
    "MODEL_PRELOAD_ENABLED":
        "Serve POST /v1/models/preload and warm the NEXT pipeline stage's "
        "model while the current one runs. Off = the endpoint still answers "
        "202 but every entry comes back 'deferred' and nothing is loaded; "
        "stages then load their model in-band on first use, as before.",
    "MODEL_PRELOAD_WARM_TTL_S":
        "How long a preload plan (and the warm leases it holds) stays alive "
        "without being touched. Re-POSTing the plan or starting any stage of "
        "the owning job restamps it, so a long job keeps its plan for free. "
        "A warm lease only makes a model ineligible for idle eviction — it "
        "never forces a load and never delays a job.",
    "MODEL_PRELOAD_VRAM_RESERVE_MB":
        "Free VRAM a preload must leave behind after loading, measured "
        "against the DRIVER's free memory (other processes on the card are "
        "invisible to our own bookkeeping). A model whose measured size "
        "would eat into this reserve is deferred, never loaded.",
    "MODEL_PRELOAD_RAM_RESERVE_MB":
        "Same reserve for CPU-placed models, measured against available "
        "system RAM.",
    "MODEL_PRELOAD_EVICT_IDLE_MODELS":
        "Let a preload evict an idle, unleased, unwarmed model of the SAME "
        "family to make room. Off = a preload that doesn't fit is simply "
        "deferred; nothing already loaded is ever disturbed.",

    # --- Speaker diarization ---
    "DIARIZATION_ENABLED":
        "Allow clients to request speaker diarization (pyannote). The "
        "pipeline loads on first use, not at startup. Needs the optional "
        "`pip install -r requirements-diarize.txt` and, for the gated "
        "models, accepted Hugging Face terms plus HF_TOKEN.",
    "DIARIZATION_MODEL":
        "pyannote pipeline id. community-1 (CC-BY-4.0) is the default; "
        "speaker-diarization-3.1 (MIT) is the alternative. Both are gated: "
        "accept the model terms with the account behind HF_TOKEN at "
        "https://huggingface.co/pyannote/speaker-diarization-community-1 "
        "(or https://huggingface.co/pyannote/speaker-diarization-3.1), "
        "else loads fail with 403.",
    "DIARIZATION_DEVICE":
        "auto follows MODEL_DEVICE (with the same fallback); cuda / cpu pin "
        "it. The pipeline holds roughly 1 GB VRAM while loaded.",
    "DIARIZATION_IDLE_TIMEOUT_S":
        "Unload the diarization pipeline after this many idle seconds, like "
        "MODEL_IDLE_TIMEOUT_S for whisper models. 0 = keep it loaded once "
        "used.",
    "DIARIZATION_EMBEDDING_BATCH_SIZE":
        "Speaker-embedding batch size. pyannote's default can spike several "
        "GB of VRAM on hour-long audio (pyannote-audio#1963); 4 keeps the "
        "peak under ~1 GB at a small wall-time cost.",
    "DIARIZE":
        "Whether a request diarizes when it does not say (the `diarize` form "
        "field overrides; lockable). Only effective while "
        "DIARIZATION_ENABLED is on.",
    "DIARIZATION_NUM_SPEAKERS":
        "Exact speaker count hint for the pipeline. Wins over MIN/MAX when "
        "set. Empty = let the pipeline decide. Clients override per request "
        "via `num_speakers`.",
    "DIARIZATION_MIN_SPEAKERS":
        "Lower bound on the speaker count. Ignored when DIARIZATION_NUM_SPEAKERS is set.",
    "DIARIZATION_MAX_SPEAKERS":
        "Upper bound on the speaker count. Ignored when DIARIZATION_NUM_SPEAKERS is set.",

    # --- Background-music separation ---
    "BGM_SEPARATION_ENABLED":
        "Allow clients to request background-music separation (UVR / "
        "MDX-Net) before transcription. Needs the optional `pip install -r "
        "requirements-bgm.txt`; the model downloads on first use.",
    "BGM_SEPARATION_UVR_MODEL":
        "UVR separation model name (.onnx implied when no extension). "
        "Downloaded to <DOWNLOAD_ROOT>/audio-separator on first use.",
    "BGM_SEPARATION_DEVICE":
        "MDX-Net runs on CPU or GPU (much faster on GPU). auto follows "
        "MODEL_DEVICE; cpu pins the separator to CPU.",
    "BGM_SEPARATION_IDLE_TIMEOUT_S":
        "Unload the separation model after this many idle seconds, like "
        "DIARIZATION_IDLE_TIMEOUT_S. 0 = keep it loaded once used.",
    "SEPARATE_BGM":
        "Whether a request separates music when it does not say (the "
        "`separate_bgm` form field overrides; lockable). Only effective "
        "while BGM_SEPARATION_ENABLED is on.",
    "DIARIZATION_ALLOWED_MODELS":
        "Allowlist of pyannote pipeline ids clients may request per-call. "
        "Both supported pipelines are listed by default; remove one to "
        "forbid it.",
    "DIARIZATION_PRELOAD":
        "Load the diarization pipeline at startup instead of on first use, "
        "so the first diarize request skips the warm-up (costs VRAM while "
        "idle).",
    "BGM_SEPARATION_ALLOWED_MODELS":
        "Allowlist of UVR separation model names clients may request "
        "per-call. Empty = only the configured BGM_SEPARATION_UVR_MODEL.",
    "BGM_SEPARATION_PRELOAD":
        "Load the separation model at startup instead of on first use, so "
        "the first separate_bgm request skips the download/warm-up.",

    # --- Translation (T2T) ---
    "TRANSLATION_ENABLED":
        "Master switch for text-to-text translation (llama.cpp GGUF "
        "models). Off = requested translation runs decline softly with a "
        "response warning and /v1/text/translations returns 403. Needs the "
        "optional `pip install -r requirements-translate.txt`.",
    "TRANSLATION_DEFAULT_MODEL":
        "GGUF model reference 'org/repo[:quant]' used when a request names "
        "no model. Empty = translation requests must name a model. Ranked "
        "picks:\n"
        "  tencent/HY-MT1.5-7B-GGUF:Q4_K_M            "
        "(Tencent, WMT25 lineage, ~5 GB)\n"
        "  mradermacher/MiLMMT-46-12B-v0.1-GGUF:Q4_K_M (Xiaomi, ~8 GB)",
    "TRANSLATION_ALLOWED_MODELS":
        "Allowlist of GGUF model refs clients may request, same semantics "
        "as ALLOWED_MODELS for whisper models: empty lets any well-formed "
        "'org/repo[:quant]' ref pass — risks unknown multi-GB downloads. "
        "Default:\n"
        "  tencent/HY-MT1.5-7B-GGUF:Q4_K_M            "
        "(Tencent, WMT25 lineage, ~5 GB)\n"
        "  mradermacher/MiLMMT-46-12B-v0.1-GGUF:Q4_K_M (Xiaomi, ~8 GB)",
    "TRANSLATION_PRELOAD_MODELS":
        "Translation models loaded eagerly at startup so the first request "
        "skips the load. Empty = load on first use.",
    "TRANSLATION_MAX_LOADED_MODELS":
        "Max GGUF translation models kept loaded at once (LRU evicts "
        "beyond this). A 7B Q4 model holds roughly 5 GB.",
    "TRANSLATION_DEVICE":
        "auto follows MODEL_DEVICE; cuda / cpu pin it. On cuda llama.cpp "
        "offloads every layer to the GPU.",
    "TRANSLATION_IDLE_TIMEOUT_S":
        "Unload a translation model after this many idle seconds, like "
        "MODEL_IDLE_TIMEOUT_S for whisper models. 0 = keep loaded once "
        "used.",
    "TRANSLATION_BATCH_SEGMENTS":
        "Segments per prompt in faithful (per-segment) mode — batched as a "
        "numbered list the model must echo back. Halved automatically when "
        "the reply's line count mismatches.",
    "TRANSLATION_PROMPT_FAMILY":
        "Prompt template family. auto = detect from the model name "
        "(hunyuan / translategemma / milmmt / seed-x, else the generic "
        "chatml prompt); custom renders TRANSLATION_PROMPT_TEMPLATE.",
    "TRANSLATION_PROMPT_TEMPLATE":
        "Custom prompt template, used when TRANSLATION_PROMPT_FAMILY is "
        "'custom'. Must contain {text} and {target_language}; optional "
        "slots: {source_language}, {context}, {glossary}.",
    "TRANSLATION_LANGUAGES":
        "Comma-separated language codes (e.g. 'en,de,fr-CA') offered as "
        "translation targets for EVERY translation model. Empty = each "
        "model's own list: the languages its prompt family supports plus "
        "those its Hugging Face model card names (known once it has loaded); "
        "a model with neither offers every language, untested.",
    "TRANSLATE_TO":
        "Comma-separated target language codes (e.g. 'en' or 'en,fr-CA') "
        "a transcription is translated into when the request does not say. "
        "Empty = translation off unless the request asks.",
    "TRANSLATION_MODEL":
        "Per-request default GGUF model ref 'org/repo[:quant]'. Empty = "
        "use TRANSLATION_DEFAULT_MODEL.",
    "TRANSLATION_CONTEXT_SEGMENTS":
        "Previous source segments prepended as context for each "
        "translation batch (families that support a context slot). 0 = no "
        "context.",
    "TRANSLATION_MAX_TARGETS":
        "Max target languages one request may ask for.",
    "TRANSLATION_MODE":
        "fluent merges consecutive segments into sentence groups before "
        "translating (better flow; the translation is redistributed across "
        "the member segments); faithful translates segment-by-segment "
        "(exact cue alignment).",
    "TRANSLATION_GLOSSARY":
        "Terminology enforced via the prompt: one 'source = target' pair "
        "per line, passed to prompt families that support reference "
        "pairs.",

    # --- Transcribe-from-URL (yt-dlp) ---
    "URL_DOWNLOAD_ENABLED":
        "Allow clients to transcribe from a pasted media link (YouTube, "
        "podcasts, direct audio/video URLs): the server downloads the audio "
        "with yt-dlp, then runs the normal pipeline. Off by default — "
        "enabling it makes the server fetch client-supplied URLs.",
    "URL_ALLOWED_EXTRACTORS":
        "Only these yt-dlp extractor keys (e.g. Youtube, Vimeo, Soundcloud; "
        "case-insensitive) may be downloaded. Empty = every dedicated "
        "extractor is allowed (the catch-all Generic extractor is governed "
        "by the two switches below, not by this list).",
    "URL_ALLOW_DIRECT_MEDIA":
        "Accept direct links to media files (URLs no dedicated extractor "
        "matches) after a capped probe confirms the response is audio/* or "
        "video/*. Narrower than URL_ALLOW_GENERIC: only confirmed media is "
        "fetched.",
    "URL_ALLOW_GENERIC":
        "Accept ANY URL via yt-dlp's Generic webpage extractor — the server "
        "fetches whatever page the client names, not just confirmed media. "
        "Internal targets stay blocked either way (every hop, including "
        "redirects and yt-dlp's own fetches, is checked against the "
        "private/loopback/metadata ranges with the resolved IP pinned), but "
        "this still widens how much of the public web one caller can aim "
        "the server at. Leave off unless you need it.",
    "URL_MAX_DURATION_S":
        "Reject linked media longer than this many seconds (checked from "
        "metadata before downloading). Default 14400 (4 h).",
    "URL_VIDEO_ENABLED":
        "Let clients also keep the VIDEO of a link (best video + best audio, "
        "merged by ffmpeg) so subtitles can be exported with the picture they "
        "belong to. Fetched after the audio, off the GPU path. Bytes are "
        "capped by MEDIA_MAX_BYTES like everything else.",
    "URL_SUBTITLES_ENABLED":
        "Let clients list and fetch a link's own subtitle tracks (the site's "
        "uploaded or original-language automatic captions) instead of "
        "transcribing: small capped GETs of the VTT/SRT text, never a media "
        "download. Rate-limited by URL_SUBTITLES_RATE_PER_MIN.",
    "URL_LANGUAGE_CHECK_ENABLED":
        "Let clients ask which language a link speaks before running it: "
        "the server downloads the audio (kept for the run that follows) — "
        "or, for a segmented HLS/DASH stream, only the segments it samples "
        "(nothing kept) — and Whisper listens to three 20 s pieces. The costliest URL route — it "
        "takes a download slot and a GPU slot; rate-limited by "
        "URL_LANGUAGE_RATE_PER_MIN.",
    "URL_DOWNLOAD_TIMEOUT_S":
        "Wall-clock ceiling for one audio download subprocess; the download "
        "is killed and the request fails past it. Default 900 (15 min).",
    "URL_VIDEO_DOWNLOAD_TIMEOUT_S":
        "Wall-clock ceiling for one VIDEO download (video is 10-50x the audio "
        "bytes, so it gets its own clock). Default 3600 (1 h).",
    "URL_PREVIEW_TIMEOUT_S":
        "Wall-clock ceiling for a metadata probe (/v1/audio/url-preview and "
        "the pre-download policy check). Default 20 s.",
    "URL_SOCKET_TIMEOUT_S":
        "Per-connection socket timeout passed to yt-dlp, so a stalled "
        "remote server fails fast instead of pinning a slot.",
    "URL_DOWNLOAD_CONCURRENCY":
        "How many URL downloads may run at once (separate from "
        "INFERENCE_CONCURRENCY — downloads are network-bound and do not "
        "hold the GPU semaphore).",
    "URL_MEDIA_DIR":
        "Directory where downloaded audio is retained briefly so the "
        "client can fetch it for local playback. Wiped on startup.",
    "URL_MEDIA_TTL_S":
        "How long a downloaded file stays fetchable via "
        "/v1/audio/url-media/{id}. The window starts when the download "
        "finishes (before transcription), so keep it comfortably longer "
        "than your slowest job. Default 3600.",
    "RETAINED_MEDIA_MAX_BYTES":
        "Byte cap on URL_MEDIA_DIR (retained audio AND video, downloaded or "
        "uploaded); oldest files are evicted first when the sum exceeds it. "
        "Size it for several videos at MEDIA_MAX_BYTES. Default 50 GB.",

    # --- Media export ---
    "MEDIA_PACKAGE_ENABLED":
        "Let clients export a video WITH its subtitle tracks: the server "
        "muxes the client's SRT files into the retained video (a link's, or "
        "one uploaded for the purpose) as soft subtitle streams — a stream "
        "copy, never a re-encode. Needs ffmpeg.",
    "MEDIA_PACKAGE_TIMEOUT_S":
        "Wall-clock ceiling for one packaging run (ffmpeg stream copy: "
        "minutes for a multi-GB file on slow disks). Default 900.",

    # --- Pipeline ---
    "PIPELINE_RULES":
        "Ordered text-cleanup rules applied to the joined transcript. Each "
        "row is a regex or named-callback rule; drag to reorder, edit, "
        "disable, or add custom rules. Reset to defaults if anything breaks. "
        "The final 'trim edges' row always runs last.",
    "PIPELINE_RULES_EXCLUDE":
        "(Per-model only) List of pipeline rule slugs to FORCE-DISABLE when "
        "this model is serving the request — even if the rule is enabled "
        "globally. Use to drop e.g. 'de-dictation-map' for German fine-tunes "
        "that already emit punctuation symbols.",
    "PIPELINE_RULES_INCLUDE":
        "(Per-model only) List of pipeline rule slugs to FORCE-ENABLE when "
        "this model is serving the request — even if the rule is disabled "
        "globally. Inverse of PIPELINE_RULES_EXCLUDE; a slug cannot appear "
        "in both lists at once.",
    "MODEL_OVERRIDES":
        "Per-model override bundle. Maps model id → override dict. Each "
        "override may set any of the per-model-overrideable fields; "
        "absent fields inherit the global default. Edited via the per-"
        "model pane of the admin UI.",
    "OVERRIDE_PROFILES":
        "Reusable per-identity config profiles. Maps profile name → override "
        "bundle (decode + streaming fields, pipeline-rule include/exclude, and "
        "a `locks` list). Users and API keys reference profiles by name; the "
        "effective config resolves per-key → per-user → profiles → per-model → "
        "global. Edited on the /settings/overrides page.",
    "ALLOW_REQUEST_OVERRIDE_PROFILE":
        "Honor a per-request `override_profile` name on the transcription and "
        "streaming endpoints. On = a request may name an OVERRIDE_PROFILES entry; "
        "it applies as the least-specific identity layer (fills only fields no "
        "per-key/per-user binding set, and can never override or unlock an "
        "admin-pinned value). Off = the request field is ignored. A per-user / "
        "per-API-key binding can narrow this (gate + allowlist) but never widen it.",
    "ALLOW_REQUEST_DECODE_OVERRIDES":
        "Honor per-request `decode_overrides` (the client's inline decode/VAD "
        "parameter tweaks). On = requests may customise decode parameters, subject "
        "to per-field admin locks. Off = every request override is ignored and "
        "reported back as `overrides_ignored`. A per-user / per-API-key binding can "
        "set this to False to disable customisation for that identity; it can only "
        "restrict, never widen, this global floor.",
    "REVISION":
        "(Per-model only) HuggingFace git revision (branch, tag, commit) "
        "to pin the model snapshot to. Empty = HEAD of default branch.",

    # --- Logging ---
    "TRACE_ENABLED":
        "Emit a multi-line trace block per transcription request. Disable "
        "on busy servers to control log volume.",

    "LOG_FILE":
        "Path to the rotating log file. Parent directory is auto-created "
        "at startup if missing.",
    "LOG_MAX_BYTES":
        "Rotate the log file when it reaches this size in bytes.",
    "LOG_BACKUP_COUNT":
        "Number of rotated log files to retain (.1, .2, …). Older files "
        "are deleted.",
    "LOG_VIEWER_INITIAL_LINES":
        "Backlog lines streamed to the /logs page on connect. When the "
        "active log has fewer lines than this (e.g. right after rotation) "
        "the viewer spills into the rotated chain (.1, .2, …) to fill the "
        "backlog. Raise on chatty TRACE_ENABLED deployments where the "
        "default leaves only a handful of requests visible.",
    "LOG_VIEWER_DOM_MAX":
        "Max number of log lines retained in the browser DOM during live "
        "tail. 0 = auto (= LOG_VIEWER_INITIAL_LINES × 4). The cap applies "
        "only to live-tail appends — \"Load older\" pagination is allowed "
        "to grow the DOM beyond it.",
    "LOG_SEGMENT_ROWS_MAX":
        "Per-segment rows the request receipt writes to the log file. The "
        "viewer can only reveal rows that were written, so this is the real "
        "ceiling on \"show more\". 0 = unlimited. A 31-minute interview is "
        "roughly 612 rows; longer files truncate with a \"(N not logged)\" "
        "note rather than silently.",
    "LOG_SEGMENT_ROWS_SHOWN":
        "Segment rows the /logs viewer shows before folding the rest behind "
        "a \"show 50 / show all\" control. Display-only — the folded rows "
        "are already present and searchable in the log file.",
    "LOG_RECEIPT_HOLD_S":
        "Idle timeout, in seconds, on a dictation receipt held open while "
        "its translation runs as a separate request. Not an absolute "
        "deadline: each progress heartbeat from the translate job restamps "
        "it, so a slow cold model load waits as long as it needs. Only a "
        "crashed, wedged or never-sent translation trips it, and the "
        "receipt is then released with a note saying why.",
    "LOG_STAGE_COLORS":
        "Colorize the receipt's per-stage sections in the /logs viewer "
        "using the same hues the app's stage rail uses, so a stage reads "
        "the same color in the log as while it ran. Disable for a "
        "monochrome log.",
    "CONSOLE_LOG_LEVEL":
        "Lowest level written to the console (stderr — what docker logs "
        "and journald capture). The log file and the /logs page always "
        "keep INFO, whatever this says. Default warning: each "
        "transcription's log block is an INFO record carrying the "
        "transcript text, so it stays out of container logs. info = the "
        "full log on the console too; debug additionally prints library "
        "debug output (console only). uvicorn's own lines follow "
        "SERVER_LOG_LEVEL. Applies immediately.",

    # --- Server ---
    "SERVER_HOST":
        "uvicorn bind address. 0.0.0.0 = listen on all interfaces "
        "(LAN-reachable); 127.0.0.1 = loopback only.",
    "SERVER_PORT":
        "uvicorn TCP port to bind. Default 8000.",
    "SERVER_WORKERS":
        "uvicorn worker processes. Keep at 1 — each worker reloads models "
        "into VRAM and multiplies GPU memory.",
    "SERVER_LOG_LEVEL":
        "uvicorn log verbosity: critical | error | warning | info | debug.",
    "MEDIA_MAX_BYTES":
        "Hard ceiling on ONE media file, in bytes: a /v1/audio/transcriptions "
        "upload (413 above it), the audio or video fetched for a link, and a "
        "video uploaded for subtitle packaging. Default 10 GB.",
    "MAX_REQUEST_BYTES":
        "Ceiling on a multipart request body, in bytes, applied from "
        "Content-Length before the body is read. Keep it above "
        "MEDIA_MAX_BYTES so uploads hit their own cap first. Other non-JSON "
        "bodies keep a 256 MiB backstop regardless. Default 10 GiB.",

    # --- Access & sessions ---
    "ADMIN_WEBUI_ALLOWED_HOSTS":
        "IP/CIDR allowlist for the admin pages (/settings, /settings/api-keys, "
        "/docs) — an admin API key is also required. Loopback (127.0.0.1, ::1) "
        "is always implicitly allowed; default is loopback only.",
    "USER_WEBUI_ALLOWED_HOSTS":
        "IP/CIDR allowlist for the user pages (/quick-config, /captures, "
        "/reports, /stats, /logs, /sev) — the per-page API key is still "
        "required. Loopback always allowed; default is OPEN (0.0.0.0/0, ::/0), "
        "so narrow it to restrict which networks may reach the pages.",
    "CORS_ALLOW_ORIGINS":
        "CORS allowlist for cross-origin browser calls to the JSON API (e.g. a "
        "third-party browser app on another origin calling this backend). Each entry "
        "is an origin like 'https://app.example.com' or 'http://192.168.1.50:8000'; "
        "'*' allows any origin (credentials then disabled). Empty (default) = CORS "
        "off. Streaming (WebSocket) is not subject to CORS. Restart required.",
    "TRUSTED_ORIGINS":
        "Extra origins accepted by the same-origin check on POST/PUT/DELETE, on "
        "top of the request's own Host. Only needed behind a reverse proxy that "
        "rewrites Host to the upstream (Nginx Proxy Manager, NPMplus, Caddy and "
        "Traefik pass it through, so nothing is needed there) — list the public "
        "origin, e.g. 'https://whisper.example.com'. Grants no cross-origin "
        "access: CORS is configured solely by CORS_ALLOW_ORIGINS. No '*'. "
        "Restart required.",
    # --- Browser sessions ---
    "SESSION_COOKIE_SECURE":
        "Mark the WebUI session/CSRF cookies 'Secure' (sent only over HTTPS). "
        "Leave OFF for plain-HTTP LAN/VPN access; turn ON when serving over "
        "HTTPS (e.g. behind a TLS reverse proxy), else login silently fails.",
    "API_KEYS_DB":
        "Path to the SQLite file holding user accounts and hashed API keys. "
        "Read at startup; a change takes effect after a restart.",
    "SESSIONS_DB":
        "Path to the SQLite file holding browser login sessions (hashed "
        "tokens, expiry). Safe to delete: everyone signs in again. Read at "
        "startup.",
    "SESSION_TTL_S":
        "Browser-session lifetime in seconds, counted from sign-in: the "
        "session cookie expires this long after login whatever the activity, "
        "and the user signs in again. Default 2592000 (30 days).",
    "SESSION_COOKIE_NAME":
        "Name of the HttpOnly session cookie. Letters, digits, '_' and '-' "
        "only. Must differ from SESSION_CSRF_COOKIE_NAME.",
    "SESSION_CSRF_COOKIE_NAME":
        "Name of the JS-readable CSRF cookie echoed back as the X-CSRF-Token "
        "header on cookie-authenticated mutations. Letters, digits, '_', '-'. "
        "Must differ from SESSION_COOKIE_NAME.",
    # --- Concurrency & request limits ---
    "TRANSLATE_MAX_INFLIGHT_PER_USER":
        "Max /v1/text/translations requests one identity (user, else API key, "
        "else client IP) may have decoding at once; further requests are "
        "refused with 429 rather than queued behind a long job. 0 = unlimited.",
    "TRANSLATE_RATE_PER_MIN":
        "Ceiling on /v1/text/translations requests per identity per 60 "
        "seconds. A loose backstop against a runaway client loop, not a quota "
        "— the in-flight cap is what protects the GPU. 0 = unlimited.",
    "STREAMING_MAX_SESSIONS_PER_USER":
        "Max simultaneous streaming sessions ONE identity may hold. Checked "
        "before the server-wide STREAMING_MAX_SESSIONS, so one client cannot "
        "take the whole pool. 0 = unlimited (only the global cap applies).",
    "URL_PREVIEW_RATE_PER_MIN":
        "Ceiling on URL-preview requests per identity per 60 seconds. Each "
        "preview makes the SERVER fetch a third-party page, so this bounds "
        "what an authenticated client can aim outbound. 0 = unlimited.",
    "URL_VIDEO_RATE_PER_MIN":
        "Ceiling on video downloads (a link run that keeps the video, or the "
        "on-demand video route) per identity per 60 seconds. Each one pulls "
        "up to MEDIA_MAX_BYTES from a third-party site. 0 = unlimited.",
    "URL_SUBTITLES_RATE_PER_MIN":
        "Ceiling on subtitle fetches (POST /v1/audio/url-subtitles) per "
        "identity per 60 seconds. Each re-probes the link and fetches up to "
        "8 tracks from the site. 0 = unlimited.",
    "URL_LANGUAGE_RATE_PER_MIN":
        "Ceiling on link language checks (POST /v1/audio/url-language) per "
        "identity per 60 seconds. Each downloads the whole audio (a "
        "segmented stream: just the sampled segments) and runs three short "
        "language detections on the GPU. 0 = unlimited.",
    "MEDIA_UPLOAD_RATE_PER_MIN":
        "Ceiling on video uploads for packaging (POST /v1/audio/media) per "
        "identity per 60 seconds — each can be MEDIA_MAX_BYTES. 0 = unlimited.",
    "MEDIA_PACKAGE_RATE_PER_MIN":
        "Ceiling on packaging requests per identity per 60 seconds. 0 = "
        "unlimited.",
    "MEDIA_PACKAGE_MAX_INFLIGHT_PER_USER":
        "Max simultaneous packaging runs ONE identity may hold (each is an "
        "ffmpeg process copying a multi-GB file). 0 = unlimited.",
    "CAPTURES_AUDIO_RATE_PER_MIN":
        "Ceiling on capture-audio fetches per identity per 60 seconds. Sized "
        "for the review UI's burst pattern (scrubbing a page of captures), "
        "not for steady-state use. 0 = unlimited.",
    "REPORTS_SUBMIT_RATE_PER_10MIN":
        "Ceiling on report submissions per identity per 600 seconds — keeps "
        "one client from flooding the reports store. 0 = unlimited.",
    "LOGIN_FAILURE_RATE":
        "Max FAILED /auth/login attempts per client host per 60 seconds "
        "before further attempts are refused; a successful login clears the "
        "host's window immediately. 0 = unlimited.",

    # --- Reports store ---
    "REPORTS_DB":
        "Path to the SQLite file holding transcription error reports. "
        "Contains plaintext dictation content — keep on an encrypted "
        "volume if sensitive.",
    "REPORTS_MAX":
        "Soft cap on the report count. On overflow, oldest closed reports "
        "(resolved/dismissed) are evicted first, then oldest open.",
    "REPORTS_RETENTION_DAYS":
        "Auto-delete reports older than this many days. Sweep runs on "
        "startup and hourly thereafter. 0 = retention sweep disabled "
        "(admin must clear manually).",
    "REPORTS_ALLOW_USER_SUBMIT":
        "Master switch for end-user (non-admin API-key) report submission. "
        "Off = only admins can submit; the button stays visible but the "
        "endpoint returns 403 for non-admin callers.",

    # --- Recent transcriptions store ---
    "RECENT_TRANSCRIPTIONS_DB":
        "Path to the SQLite file holding the persistent /quick-config "
        "trace panel + /stats \"Recent transcriptions\" widget data. "
        "Plaintext dictation content — keep on an encrypted volume if "
        "sensitive.",
    "RECENT_TRANSCRIPTIONS_MAX":
        "Hard row-count cap. 0 = unbounded (TTL-only pruning). Lazy "
        "pruning runs every RECENT_TRANSCRIPTIONS_PRUNE_EVERY inserts, "
        "so on-disk count can briefly exceed this by up to PRUNE_EVERY "
        "rows before the next sweep.",
    "RECENT_TRANSCRIPTIONS_RETENTION_DAYS":
        "Auto-delete entries older than this many days. 0 = TTL "
        "disabled (count-cap only). Combined with the row cap: "
        "whichever bound is tighter wins.",
    # --- Server jobs ---
    "JOBS_ENABLED":
        "Keep a durable record of every batch run posted with a "
        "progress_id — status and the verbatim result — behind "
        "GET/DELETE /v1/jobs*, so a client that lost its connection can "
        "re-attach and fetch the run. Off = 403 on /v1/jobs*, no rows.",
    "JOBS_DB":
        "Path to the SQLite file holding the job rows. Carries transcript "
        "text — keep on an encrypted volume if sensitive. Read at startup.",
    "JOBS_TTL_S":
        "Seconds a job (and its stored result) stays fetchable after it "
        "finishes. Default 259200 (72 h).",
    "JOBS_MAX_ROWS":
        "Row cap: the newest rows are kept, a running one is never "
        "evicted. Swept lazily on insert and hourly.",
    "JOBS_MAX_BYTES":
        "Cap on the stored result bytes across all rows; the oldest "
        "finished results are dropped first. 0 = no byte cap.",
    "JOBS_RATE_PER_MIN":
        "GET/DELETE /v1/jobs* requests per identity per 60 s (a "
        "re-attached client polls once a second). 0 = unlimited.",
    # --- Usage statistics ---
    "USAGE_DB":
        "Path to the SQLite file holding the usage ledger (per-job numbers "
        "and hourly rollups, no transcript text). Read at startup.",
    "CLIENT_SETTINGS_DB":
        "Path to the SQLite file holding the desktop app's synced settings "
        "blobs, one per user and profile. Read at startup.",
    "USAGE_RETENTION_DAYS":
        "Auto-delete hourly usage rollup rows (requests, words, audio, "
        "stages, dictation outcomes) older than this many days. 0 = keep "
        "forever — the rollup is tiny and lifetime totals stay complete.",
    "USAGE_JOBS_RETENTION_DAYS":
        "Auto-delete per-job usage rows (one per dictation session, file, "
        "URL or text translation, with their stage detail) older than this "
        "many days. 0 = keep forever.",
    "USAGE_APP_RETENTION_DAYS":
        "Auto-delete the per-app dictation rollup (which program each "
        "dictation was typed into) older than this many days. Names "
        "programs on the user's machine, so it defaults shorter than the "
        "rest. 0 = keep forever.",
    "USAGE_UNREPORTED_AFTER_H":
        "Hours after which a dictation session the desktop app never "
        "reported an outcome for (closed, crashed, or reporting switched "
        "off) is counted as 'unreported' in the dictation breakdown.",

    "RECENT_TRANSCRIPTIONS_PAGE_SIZE":
        "Number of entries the browser fetches per page on "
        "/quick-config (initial load + each \"Load older\" click). "
        "Also clamps the server-side LIMIT.",
    "QUICK_CONFIG_MAP_COLLAPSE_AFTER":
        "How many of the newest spoken-symbol (callback:map) entries the "
        "/quick-config rule editor shows before collapsing the rest behind a "
        "\"show older\" toggle. Also served to the desktop client. 0 = show all.",
    "QUICK_CONFIG_WORD_SUGGESTIONS_MAX":
        "How many recently-transcribed word/phrase suggestions the spoken-symbol "
        "(callback:map) key field offers as autocomplete (on /quick-config and "
        "the desktop Dictionary via /v1/recent-words). Scoped per user. "
        "0 = disabled.",
    "RECENT_TRANSCRIPTIONS_PRUNE_EVERY":
        "Lazy-prune cadence — every Nth insert runs a single DELETE "
        "that enforces both the row cap and the TTL. 0 disables lazy "
        "pruning entirely (rows accumulate until manual /clear).",
    "STATS_RECENT_TRANSCRIPTIONS_COUNT":
        "/stats dashboard \"Recent transcriptions\" widget row count. "
        "The widget is intentionally a small ticker — bumping this "
        "past ~50 makes it scroll awkwardly without adding signal.",
    "STATS_OWN_SCOPE_SHOW_SYSTEM_METRICS":
        "Own-scope users see machine cards. Users whose /stats page scope is "
        "\"own\" normally get only their own jobs and usage plus a coarse "
        "server block (GPU busy/idle, VRAM headroom). On = they also see the "
        "full machine cards (GPU/CPU/RAM/process/latency/endpoints/5xx/"
        "models), which reveal when other people run jobs — fine for a "
        "trusted household box. Admins and \"all\" scope are unaffected.",
    "STATS_SYSTEM_METRICS_DB":
        "Path to the SQLite file holding the system-metrics history (GPU / "
        "CPU / RAM readings for the /stats charts). Rolling telemetry, no "
        "dictation content.",
    "STATS_SYSTEM_METRICS_SAMPLE_S":
        "Seconds between system-metrics readings (GPU utilisation / VRAM / temperature, "
        "CPU, RAM, GPU-busy share) kept for the /stats history charts. 10 s "
        "keeps a week at ~60k rows; larger values thin the curves.",
    "STATS_SYSTEM_METRICS_RETENTION_DAYS":
        "Days of machine samples to keep for the /stats history charts "
        "(pruned hourly). 0 keeps them forever.",

    # --- Captures (fine-tuning data store) ---
    "CAPTURES_RECORDING_ENABLED":
        "Master switch for capturing audio + word-timestamps next to each "
        "transcription, for use as Whisper fine-tuning training data. "
        "Default OFF — voice recordings are biometric-grade personal data "
        "and persist on disk in plaintext (encrypt the volume). Per-model "
        "WORD_TIMESTAMPS_ENABLED=False overrides this for that model: "
        "capture is skipped to avoid corrupting alignment data on models "
        "(e.g. primeline / tnfru) where DTW is broken.",
    "CAPTURES_DB":
        "Path to the SQLite file holding capture metadata + word "
        "timestamps + admin corrections. Audio files live separately "
        "under CAPTURES_DIR; this DB references them by relative path.",
    "CAPTURES_DIR":
        "Filesystem root for captured audio files. Files use a 4-char "
        "fanout (<dir>/<id[0:2]>/<id[2:4]>/<id>.<ext>) to keep directory "
        "sizes modest. Raw voice audio on disk — encrypt the volume if "
        "sensitive.",
    "CAPTURES_MAX":
        "Soft cap on capture row count. On overflow, oldest rows are "
        "evicted in priority order: dismissed → audio_missing → reviewed "
        "→ new → ready (training data is protected).",
    "CAPTURES_MAX_MB":
        "Soft cap on total audio bytes (sum of files under CAPTURES_DIR, "
        "in megabytes). Eviction policy mirrors CAPTURES_MAX.",
    "CAPTURES_RETENTION_DAYS":
        "Auto-delete captures older than this many days. 0 = retention "
        "disabled (admin must clear manually). Sweep runs on startup and "
        "hourly thereafter.",
    "CAPTURES_RECORDING_SAMPLE_RATE":
        "Fraction of eligible transcription requests to capture, in "
        "[0.0, 1.0]. 1.0 captures every eligible request; lower values "
        "are useful when you have a lot of traffic and only need a "
        "representative sample for fine-tuning.",
    "CAPTURES_RECORDING_MIN_DURATION_S":
        "Skip capture for clips shorter than this. Filters out false "
        "starts and silence pings that VAD almost fully suppresses.",
    "CAPTURES_RECORDING_MAX_DURATION_S":
        "Skip capture for clips longer than this. Whisper fine-tuning "
        "prefers ≤30s samples; long clips can still be captured for "
        "later segmentation via the stored segments metadata, but "
        "very long clips are usually not worth the disk cost.",
    "CAPTURES_RECORDING_AUDIO_BYTES_HARD_LIMIT":
        "Pre-transcribe upload-size guard. Captures eligibility roll is "
        "skipped for uploads larger than this many bytes, even when "
        "sampling would otherwise pass.",
    "CAPTURES_PIPELINE_RULES_EXCLUDE":
        "Set of PIPELINE_RULES slugs to SKIP when computing each "
        "capture's `text_for_training` (the column /captures shows and "
        "the export emits). All other PIPELINE_RULES still run. Default "
        "skips `de-dictation-map` + `capitalize-after-terminator` so the "
        "stored training text matches Whisper's raw output under "
        "SUPPRESS_CHARS — \"Komma\"/\"Punkt\" stay as words; sentence-"
        "internal lowercase preserved. /transcribe runtime output is "
        "unaffected (it still applies the full pipeline). Edit + run "
        "Reprocess all to apply changes to existing captures.",
    "CAPTURES_VAD_TRIM_ENABLED_FOR_SAMPLES":
        "When True, EVERY member of a sample is silence-trimmed via Silero "
        "VAD before merge_wavs() concatenates them: outer edges down to "
        "CAPTURES_VAD_MARGIN_SAMPLE_EDGE_MS and internal gaps capped at "
        "CAPTURES_VAD_MARGIN_SAMPLE_INTERNAL_MS. Removes the multi-second "
        "dead air that used to stack up at member joins (member i trailing "
        "+ gap + member i+1 leading silence). Applies to newly created / "
        "re-merged samples; mitigates the hallucination failure mode in "
        "arXiv:2505.12969 (Calm-Whisper).",
    "CAPTURES_VAD_MARGIN_SAMPLE_EDGE_MS":
        "Per-member sample trim: silence kept on each member's outer edges "
        "(default 300 ms) so tight VAD boundaries don't clip word onsets. "
        "Lower for tighter merges; raise if you hear clipped starts/ends.",
    "CAPTURES_VAD_MARGIN_SAMPLE_INTERNAL_MS":
        "All internal silence in a merged sample, in ms (default 300): the "
        "gap inserted BETWEEN members (normalized — added if the members' "
        "trimmed edges are below this, trimmed if above) and the cap on "
        "pauses WITHIN a member. The single inter-utterance silence knob.",
    "CAPTURES_SAMPLE_MIN_DURATION_S":
        "Minimum length of a finished training sample, in seconds "
        "(default 1.0). A junk floor that discards near-empty samples; the "
        "proposer packs the bulk toward the target so this mainly bounds "
        "single-capture samples. Must be ≤ the proposer target.",
    "CAPTURES_SAMPLE_MAX_DURATION_S":
        "Hard maximum length of a finished sample, in seconds (default "
        "29.9; must be ≤ 30, Whisper's window). The single source of truth "
        "for the merge, the pre-merge validation, the merge estimate, and "
        "the proposer cap. Must be ≥ the proposer target.",
    "CAPTURES_SAMPLE_JOIN_STRATEGY":
        "How member transcripts concatenate in a sample: 'space' (single "
        "space) or 'period_space' ('. '). Applies to every new or "
        "regenerated sample.",
    "CAPTURES_PROPOSER_TARGET_S":
        "Length the auto-proposer packs samples toward, in seconds "
        "(default 26). The fill-score peak; keep ≥1 s below the max so the "
        "proposer doesn't camp at the rejection edge. Must sit between the "
        "sample min and max.",
    "CAPTURES_PROPOSER_SESSION_GAP_S":
        "Captures more than this many seconds apart start a new session "
        "bucket for proposal grouping (default 1800).",
    "CAPTURES_PROPOSER_DUP_THRESHOLD":
        "Reject pairing two captures in one proposal when their transcript "
        "similarity ratio exceeds this (0–1, default 0.85) — a near-"
        "duplicate / echo guard.",
    "CAPTURES_PROPOSER_MAX_PROPOSALS":
        "Maximum number of merge proposals returned per request "
        "(default 20).",
    # --- Live streaming (WebSocket dictation) ---
    "STREAMING_ENABLED":
        "Enable the live dictation WebSocket at /v1/audio/transcriptions/stream. "
        "Off → the endpoint refuses connections (batch /transcribe is unaffected).",
    "STREAMING_MAX_SESSIONS":
        "Max simultaneous streaming sessions; further connections are refused. "
        "Bound to your GPU's real concurrent capacity. A single client is "
        "additionally capped by STREAMING_MAX_SESSIONS_PER_USER, which is "
        "checked first.",
    "STREAMING_IDLE_TIMEOUT_S":
        "Close a live-dictation connection that sends no audio for this many "
        "seconds, freeing its slot (bounds idle/abandoned connections). 0 = off. "
        "Won't cut a normal session (the client streams continuously).",
    "STREAMING_WS_PING_INTERVAL_S":
        "WebSocket keepalive ping interval (s), passed to uvicorn — the server "
        "pings clients this often so a dead connection is detected. 0 = no "
        "keepalive pings (restart to change).",
    "STREAMING_WS_PING_TIMEOUT_S":
        "WebSocket keepalive timeout (s): drop the socket if a ping gets no pong "
        "within this long. Generous, since the decode runs off the receive loop. "
        "0 = disable (restart to change).",
    "INFERENCE_CONCURRENCY":
        "Shared cap on concurrent GPU decodes across BOTH streaming and the batch "
        "/transcribe route — prevents oversubscribing the GPU (restart to change).",
    "STREAMING_PARTIAL_MODEL":
        "Optional fast model for the live partial loop (e.g. a turbo-German CT2 "
        "id). Empty = use the request's model for partials too (lowest VRAM).",
    "STREAMING_PARTIAL_BEAM":
        "Beam width for partial decodes. 5 is ~as fast as greedy (encoder-bound) "
        "but more stable run-to-run. Finals use BEAM_SIZE.",
    "STREAMING_PARTIAL_TEMPERATURE":
        "Sampling temperature for partial decodes (single value, 0–1). Kept at 0.0 "
        "so partials never trigger a fallback re-decode mid-stream; finals use the "
        "per-model TEMPERATURE ladder.",
    "STREAMING_PARTIAL_CONDITION_ON_PREVIOUS_TEXT":
        "Condition partial decodes on previously committed text. Off by default — "
        "many German finetunes loop into repeated punctuation on short, growing "
        "partial buffers. Finals use the per-model CONDITION_ON_PREVIOUS_TEXT.",
    "STREAMING_VAD_BACKEND":
        "Endpointing backend: 'auto' (Silero if installed, else energy), 'silero' "
        "(noise-robust, needs the silero-vad package), or 'energy' (pure RMS).",
    "STREAMING_VAD_THRESHOLD":
        "Silero speech-probability cutoff (0–1). Speech ends at threshold−0.15 "
        "(built-in hysteresis). Lower for quiet speakers.",
    "STREAMING_GATE_RMS_DBFS":
        "Skip inference when the buffer is quieter than this (dBFS) — a backstop "
        "against silence/noise hallucinations. Typical −42.",
    "STREAMING_PARTIAL_INTERVAL_MS":
        "Partial cadence: new audio accumulated before re-decoding (ms). 1000 is "
        "the validated German sweet spot (~4.4 s stabilization latency).",
    "STREAMING_GATE_MIN_SPEECH_MS":
        "Minimum speech in the buffer before any decode runs (ms) — sub-500 ms "
        "buffers hallucinate.",
    "STREAMING_FINAL_DROP_MIN_AVG_LOGPROB":
        "Post-decode anti-hallucination guard: drop a FINAL segment only when its "
        "average log-probability is below this AND it fell through the temperature "
        "ladder. Lower (e.g. -10) to effectively disable.",
    "STREAMING_FINAL_DROP_TEMPERATURE":
        "Temperature at/above which a low-confidence FINAL segment is treated as a "
        "failed decode and dropped (paired with the log-prob floor). Requiring "
        "both signals avoids discarding genuine quiet speech.",
    "STREAMING_FINAL_CONDITION_ON_PREVIOUS_TEXT":
        "Condition the FINAL decode's later windows on its earlier output. Off "
        "by default: a leftover sub-second window after the last word would "
        "otherwise see the rolling prompt + the utterance's own text and echo "
        "it verbatim into the transcript. Cross-utterance context via "
        "initial_prompt is unaffected. Batch keeps CONDITION_ON_PREVIOUS_TEXT.",
    "STREAMING_FINAL_BEST_OF":
        "Candidates a FINAL decode's retry attempts (temperature > 0) write in "
        "parallel. The attempt only ends when the slowest candidate ends, so "
        "one candidate stuck in a loop makes a short correct answer wait 10+ s. "
        "1 (default) avoids that; raise towards BEST_OF for slightly more "
        "robust retries at that latency risk. The first attempt (BEAM_SIZE) is "
        "unaffected. Batch keeps BEST_OF.",
    "STREAMING_TAIL_TRIM_PAD_MS":
        "Audio kept (ms) after the last detected speech when trimming the "
        "trailing endpointer silence/noise off the FINAL decode buffer — "
        "removes the non-speech tail Whisper hallucinates into. 0 = no trim. "
        "Default 300.",
    "STREAMING_VAD_INNER_SILENCE_MS":
        "Inner silence gate (ms): a pause this long triggers a boundary partial "
        "without finalizing. Spans German sub-clause pauses (~700).",
    "STREAMING_VAD_OUTER_SILENCE_MS":
        "Outer silence gate (ms): end-of-speech silence that finalizes the "
        "utterance and runs post-processing. Tune per dictation habit (~1200).",
    "STREAMING_HARD_BREAK_SILENCE_MS":
        "Silence (ms) that ends the whole grouping and starts a fresh document "
        "mid-connection — bounds long latch sessions and makes pauses act as "
        "paragraph breaks (resets prompt + committed context). 0 = off (~5000).",
    "STREAMING_HARD_BREAK_SEPARATOR":
        "Text the client types between documents at a hard break: '' = nothing, "
        "' ' = space, '\\n' = newline.",
    "STREAMING_FORCED_COMMIT_S":
        "Hard cap (s) on continuous speech before a forced finalize — keeps the "
        "buffer inside Whisper's 30 s receptive field. Must be < 30.",
    "STREAMING_BUFFER_TRIM_S":
        "Trim the audio buffer once it grows past this (s), at a committed "
        "word boundary, to bound decode cost.",
    "STREAMING_BUFFER_TRIM_KEEP_S":
        "Audio retained (s) after a trim, as left-context for the next decode.",
    "STREAMING_MAX_BUFFER_S":
        "Last-resort ceiling (s) on one utterance's audio buffer: past it the "
        "utterance is force-finalized. Deliberately generous (24x the forced "
        "commit) so real dictation never reaches it — it only catches a buffer "
        "filled by silence when the trim can't run (VAD flicker on a noisy mic).",
    "STREAMING_PROMPT_WORDS":
        "Confirmed words carried across utterances as initial_prompt for "
        "cross-sentence context (drug names, terminology). ~200.",
}
