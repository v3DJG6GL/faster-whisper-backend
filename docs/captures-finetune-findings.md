# /captures → Whisper fine-tuning — research findings

German-only dictation. Swiss-German dropped (not needed). No code changes — this is a future-plan reference document.

---

## Master decisions

| # | Topic | Decision | Status |
|---|---|---|---|
| 1 | Singleton vs group field-set divergence | Unify to one schema so singletons and groups emit the same keys; missing values become `null` / `""` / `0`. | Future plan |
| 2 | `language` field empty for groups | Capture `language` at trace time (already known: `de (prob=1.00)`), persist on group + singleton, emit in export. | Future plan |
| 3 | `is_stale=true` / `is_locked=true` leak into export | Filter out at export by default (rolled into #4). | Future plan |
| 4 | Export-scope principle | **Only `status=ready` is ever exported.** No `audio_missing`, no `is_stale`/`is_locked`, no `new`/`reviewed`/`dismissed` by default. Status is the quality-gate principle. | Future plan |
| 5 | Group join uses `\n` | Drop `newline` join strategy entirely; force `space` or `". "`. Whisper never emits literal `\n` in continuous speech. | Future plan |
| 6 | Data volume per model size | Verified — see "Data quantity" section. Numbers drop vs Swiss-German case because no acoustic shift, only vocab. | Informational |
| 7 | Silence trim | Apply at group-merge time (Silero VAD) + per-singleton trim button so reviewers see and hear exactly what Whisper will be trained on. | Future plan |
| 9 | Punctuation strategy | Word-form ("Komma"/"Punkt" stay as German words), NO symbol punctuation, preserve German casing as Whisper raw produces it (proper nouns capitalized; "ich"/"wie" lowercase because no terminator cue). | Future plan |
| 14.2.1 | Rule-deselect mechanism | Applied at storage time (not export); captures-specific config separate from `/transcribe`. Default-deselect: `dictation-map` + `capitalize-after-terminator`. Other rules keep applying. | Future plan |
| 14.2.2 | Capitalization claim | Earlier claim was wrong. Raw Whisper output doesn't capitalize "ich"/"wie" because there's no terminator cue. Training text matches raw exactly. | Corrected |
| 14.2.3 | Mixed-corpora warning | Dropped — dictation-only by design. | — |
| 15 | Alternative self-hostable ASR models | Survey below. Stay on faster-whisper, switch fine-tune base to `primeline/whisper-large-v3-turbo-german` (Apache 2.0, ~2.6 % de WER). Optional Stage-2 medical-term post-processor. | Future plan |
| Dropped | Swiss-German | Not needed — dictation is in Standard German. Flurin17, recapp, STT4SG-350, SwissDial drop out of plan. | — |

---

## Data quantity for German-only medical dictation

Without Swiss-German acoustic shift, the threshold drops — only medical vocabulary adaptation needed.

| Model | Measurable WER drop (≥ 2 pp) | Production-grade | LoRA sufficient up to | Notes |
|---|---|---|---|---|
| `primeline/whisper-large-v3-turbo-german` (809 M) | 5–10 h of medical dictation | 30–80 h | ~80 h before full FT becomes worthwhile | Best default. Apache 2.0, ~2.6 % de WER baseline, 4 GB VRAM, drop-in for CTranslate2 stack. |
| `primeline/whisper-large-v3-german` (1.55 B) | 8–15 h | 50–150 h | ~100 h | Bigger headroom for fine-tune on medical vocab. Slower at inference than turbo. 6–10 GB VRAM. 3.00 % WER on Common Voice de. |
| `primeline/distil-whisper-large-v3-german` (756 M) | 5–10 h | 30–80 h | ~80 h | Throughput-optimised; 1.5 GB, near-large quality. |
| `nvidia/canary-1b-v2` (978 M) | 15–25 h | 80–150 h | ~60 h (NeMo LoRA) | Only consider if FLEURS-de < 3 % is mission-critical — requires full stack rebuild to NeMo. |

**Flag:** below ~2 h of data, fine-tuning *degrades* tiny/base models per [ml6 Dutch sweep](https://blog.ml6.eu/fine-tuning-whisper-for-dutch-language-the-crucial-role-of-size-dd5a7012d45f). Modal's [domain-vocab tutorial](https://modal.com/docs/examples/fine_tune_asr) shows lexical bias adapts in hours, not days, because medical vocabulary is decoder-side bias, not acoustic.

---

## Punctuation strategy (decision #9 detail)

The current export sources the post-pipeline `final` text, which has converted symbols (`,` `.` `?` `!` etc.). This creates a train/inference distribution mismatch:

- **At inference:** `SUPPRESS_CHARS=",.:;-?!"` blocks those tokens. Whisper physically cannot emit them. It outputs "Komma"/"Punkt" etc., which the post-pipeline converts.
- **If trained with symbols in `text`:** the model learns to emit those tokens at sentence boundaries. At inference, they're suppressed; the model falls back to the second-best token — typically a wrong word, not the harmless dictated word.

**Correct training data:** word-form punctuation that matches raw Whisper output. Example:

```
Hansueli und Peter sind gerade dort drüben Punkt ich weiß nicht was ich dazu denken soll Ausrufezeichen ich bin gerade etwas müde Punkt Neue Zeile wie geht es dir heute so Fragezeichen
```

Note "ich"/"wie" lowercase because Whisper raw output has no terminator cue. Proper nouns and German common nouns (including "Komma", "Punkt") stay capitalized — that's natural Whisper output.

### Implementation (decision #14.2.1)

- Rule deselection applied at storage time, not export.
- Captures-specific config separate from runtime `/transcribe` (which keeps all rules).
- Default-deselect: `dictation-map` + `capitalize-after-terminator`.
- All other rules (ß→ss, ẞ→SS, digit-range normalization, dedup-space, edge-trim) keep applying.
- Reviewers see training-form text on /captures page; chips operate against training form; export reads `text` directly.
- Configurable so admins can re-enable specific rules to experiment.

### Why not "use raw" or "reverse-map symbols → words"

- **Raw lacks beneficial corrections** (typo fixes, character normalization, etc.).
- **Reverse-map is not bijective.** 12 symbols have many-to-one source words. Examples: `!` ← "Ausrufezeichen" or "Ausrufungszeichen"; `;` ← "Semikolon" or "Strichpunkt"; `-` ← "Bindestrich" or "Trennstrich" or "Minus" or "Minuszeichen". Any reverse choice introduces synthetic word-choices not what the speaker actually said.

### Honest gap

No published Whisper fine-tune A/B-tests word-form-keep vs strip-all training data for a spoken-punctuation dictation workflow. The strategy is logically sound and matches Whisper's raw output distribution exactly, but has not been independently validated for this exact setup. **Run a ~10 h LoRA ablation both ways before committing at full scale.**

---

## #15 — Self-hostable ASR alternatives survey

### Tier 1 — drop-in Whisper bases (your current stack)

| Model | License | German WER | Medical | Size / VRAM | Notes |
|---|---|---|---|---|---|
| [primeline/whisper-large-v3-turbo-german](https://huggingface.co/primeline/whisper-large-v3-turbo-german) | Apache 2.0 | ~2.6 % (vendor) | No | 809 M, ~4 GB | **Recommended default fine-tune base.** Drop-in for faster-whisper / CTranslate2 — just swap the model weights. Best license × WER × your-stack match. |
| [primeline/whisper-large-v3-german](https://huggingface.co/primeline/whisper-large-v3-german) | Apache 2.0 | 3.00 % CV-de | No | 1.55 B, ~6–10 GB | Larger; more headroom for fine-tune on medical vocab. Slower at inference than turbo. |
| [primeline/distil-whisper-large-v3-german](https://huggingface.co/primeline/distil-whisper-large-v3-german) | Apache 2.0 | near-large | No | 756 M, 1.5 GB | Latency/throughput sweet spot. Half size, near-large quality. Good for batch export processing or concurrent dictation. |

### Tier 2 — Whisper-alternative architectures

| Model | License | German WER | Medical | Size / VRAM | Notes |
|---|---|---|---|---|---|
| [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) | CC-BY-4.0 | **4.4 % FLEURS-de**, 7.27 % MLS-de | No | 978 M, ~6 GB | Best general-German numbers across alt. architectures. Requires NeMo / Triton stack — not drop-in. Worth switching only if FLEURS-de < 3 % is mission-critical. |
| [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | CC-BY-4.0 | 5.04 % FLEURS-de, 4.84 % CoVoST-de | No | 0.6 B, ~2 GB | Streaming-capable RNNT/TDT. Lowest VRAM in tier 2. 25 EU languages. Same NeMo-rebuild cost as Canary. |
| [ibm-granite/granite-speech-3.3-8b](https://huggingface.co/ibm-granite/granite-speech-3.3-8b) | Apache 2.0 | mean 5.74 OASR (DE not isolated) | No | 9 B, BF16 ~18 GB | Conformer encoder + Granite-3.3-8b LM. Heaviest in tier 2. Useful if you also want integrated LM rescoring in the same model. |
| [facebook/seamless-m4t-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large) | Seamless license (mixed restrictions) | ~7.53 % de (third-party) | No | 2.3 B, ~12 GB | Multilingual translation + ASR. License has non-commercial elements — review before SaaS use. Higher WER than tier 1. |
| [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | MIT / Phi | 6.14 % EN avg. OASR (DE not isolated) | No | 5.6 B, ~10–12 GB | New (Feb 2025). Speech + vision + text multimodal. Topped OASR leaderboard for English; German numbers not yet broken out publicly. |

### Tier 3 — medical-domain specialists

| Model | License | German WER | Medical | Size / VRAM | Notes |
|---|---|---|---|---|---|
| [google/medasr](https://huggingface.co/google/medasr) | HAI-DEF (CC-BY-4.0 docs, Apache code) | **n/a — English only** | Yes (~5 000 h physician dictation: radiology, internal, family) | 105 M Conformer, ~8 GB | **Cannot use for German.** Model card explicitly states all training audio is US English; only English-accent adaptation is suggested. Best-in-class for English clinical (4.6 % WER on RAD-DICT vs Whisper 25.3 %), irrelevant for your stack. |
| [leduckhai/MultiMed](https://huggingface.co/leduckhai/MultiMed) | Apache 2.0 | not published | Yes — 150 h multilingual incl. German | Whisper-medium class | First multilingual medical ASR. Research scale, modest hours. Worth an ablation as a secondary base or as a medical-vocabulary seed for LoRA on top of primeline. |

### Tier 4 — commercial on-prem (paid licenses)

| Model | License | German WER | Medical | Size / VRAM | Notes |
|---|---|---|---|---|---|
| Speechmatics on-prem Medical Model | Commercial (licensed container) | 7 % EN medical, 96 % medical keyword recall (DE not separately quoted) | **Yes — explicit German medical model** (> 2 B words medical training) | GPU container | Closest "drop-in production" for medical German. On-prem container deployment. Commercial pricing. Worth a trial if you want fastest time to medical-quality output. |
| NVIDIA Riva (NIM) | NVAIE (paid) | not split by lang publicly | No (general) | GPU-bound, A10+ | Container-based; needs NVIDIA AI Enterprise license. Production-grade serving on top of NeMo models. |
| Deepgram / AssemblyAI self-host | Commercial | DE supported; not split publicly | Generic (Deepgram has "Medical" tier in EN) | NVIDIA GPU required | Self-host containers from API vendors. Less mature than Speechmatics for German medical. |

### Tier 5 — dormant / out-of-scope (do not adopt)

| Model | License | Why not |
|---|---|---|
| VOSK DE 0.21 | Apache 2.0 | Kaldi-era hybrid; CPU-only operation. Behind Whisper-v3-turbo in 2024 medical study. Significantly worse than current Whisper bases on every published German benchmark. |
| Coqui STT (German) | MPL-2.0 (community fork) | Company shut down Jan 2024; models frozen at DeepSpeech-era quality. Don't build on this. |
| Nuance Dragon Medical One | Commercial | **Cloud-only now** (Azure DE: Berlin/Frankfurt). On-prem Practice Edition is legacy. Out-of-scope for self-hosting. |
| jonatasgrosman/wav2vec2-large-xlsr-53-german | Apache 2.0 | 8.74 % CV6.1 w/ LM. Lower SOTA than any 2024+ Whisper alternative. Useful only if VRAM is extremely tight (317 M params). |
| Picovoice Cheetah / Leopard | Commercial | On-device SDK (mobile-class). Not built for medical dictation quality. |
| Mozilla DeepSpeech | MPL-2.0 | Pre-Whisper era; no longer competitive. |
| Aalto-University aalto-speech | BSD-3 | Kaldi-era; last meaningful release pre-2022. |

---

## Honest gaps

| Gap | Detail |
|---|---|
| Google MedASR language coverage | **English-only.** Despite the medical-domain advantage, the model card explicitly states all training audio is US English. Not usable for German medical dictation without re-training the entire model on German medical data — which their provided notebook does not target. |
| Charité 2026 German medical benchmark | [arXiv:2601.19945](https://arxiv.org/abs/2601.19945) evaluated 29 ASR systems on simulated German anamnesis dialogues, best WER < 3 % — but it's an evaluation paper, no new model shipped. Useful as a reference WER floor for benchmarking. |
| MedASR-style brace tokens (`{period}`, `{comma}`, `{paragraph}`) | Confirmed via Google HAI-DEF forum (staff Ke_Wu cites training-data frequencies). Works because Google had transcriptionists type those tokens. **Do not copy this scheme** for Whisper — would require custom tokenizer surgery and brace-annotated training data at scale. |
| Word-form punctuation A/B benchmark | No Whisper fine-tune in the wild has A/B-tested word-form-keep vs strip-all training data for a spoken-punctuation dictation workflow. The strategy is logically sound and matches Whisper's raw output distribution exactly, but has not been independently validated. **Run a ~10 h LoRA ablation both ways** before committing at full scale. |
| `suppress_tokens` at training | Verified inference-only. Standard HF practice: set `model.config.suppress_tokens = []` before fine-tuning so the model learns freely; suppression at inference is a separate, decoupled layer. Your inference setup is therefore unaffected by training data choices. |

---

## Recommended hybrid path

### Stage 1 — Acoustic transcription (keep faster-whisper)

| Step | Detail |
|---|---|
| Base model swap | Switch from current base to `primeline/whisper-large-v3-turbo-german` (Apache 2.0, ~2.6 % de WER, ~4 GB VRAM). |
| Fine-tune approach | PEFT/LoRA on /captures medical dictation corpus. Start ~10 h, scale to 30–80 h for production-grade. Above ~150 h consider full FT. |
| Pipeline retention | Keep existing PIPELINE_RULES + post-processor. Captures-specific rule deselect (`dictation-map` + `capitalize-after-terminator`) produces training-form text at storage time. |

### Stage 2 — Domain post-processor

| Step | Detail |
|---|---|
| Medical dictionary normalizer | Add German medical dictionary normaliser (ICD-10-GM, ATC, drug brand names) after the post-pipeline. |
| Optional LLM rescoring | Route hypotheses through Granite-3.3-8b-instruct (Apache 2.0, German-fluent) with a constrained medical prompt for abbreviation expansion + casing. Matches what the Charité benchmark and MultiMed papers find consistently wins on medical-term recall *without* rebuilding the acoustic model. |

### When to revisit

| Trigger | Action |
|---|---|
| FLEURS-de < 3 % WER becomes mission-critical | Evaluate `nvidia/canary-1b-v2` on a NeMo pipeline. Cost: full stack rebuild. Gain: ~1–2 pp WER improvement. |
| Commercial budget exists and time-to-medical-quality is critical | Trial Speechmatics on-prem German Medical Model. Pre-trained on >2 B words medical data; no in-house fine-tune needed. |

---

## Open future-plan items (consolidated)

1. Unify singleton + group manifest schemas.
2. Persist + emit `language` for groups (already known at trace: `de (prob=1.00)`).
3. Export only `status=ready` (principle: status is the quality gate; drop all non-ready states by default — `audio_missing`, `is_stale`, `is_locked`, `new`, `reviewed`, `dismissed`).
4. Drop `newline` group-join strategy; force `space` or `". "`.
5. Silence-trim during group merge (Silero VAD) + per-singleton manual trim button so reviewers see/hear exactly what Whisper will be trained on.
6. Captures-specific pipeline-rule deselect config (default-skip: `dictation-map` + `capitalize-after-terminator`), applied at storage time so reviewers see training-form text.
7. Run a ~10 h LoRA ablation comparing word-form-keep vs strip-all training data on real corpus to validate Strategy A.
8. **Switch fine-tune base to `primeline/whisper-large-v3-turbo-german`** (Apache 2.0, ~2.6 % de WER, 4 GB VRAM, drop-in for CTranslate2).
9. Optional Stage-2 medical-term/casing post-processor (dictionary or LLM such as Granite-3.3-8b-instruct).
10. If FLEURS-de < 3 % becomes mission-critical: evaluate `nvidia/canary-1b-v2` on NeMo (full stack rebuild).

---

## Sources

- [primeline/whisper-large-v3-turbo-german](https://huggingface.co/primeline/whisper-large-v3-turbo-german)
- [primeline/whisper-large-v3-german](https://huggingface.co/primeline/whisper-large-v3-german)
- [primeline/distil-whisper-large-v3-german](https://huggingface.co/primeline/distil-whisper-large-v3-german)
- [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) — 4.4 % FLEURS-de
- [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — 5.04 % FLEURS-de, 25 EU langs
- [IBM Granite Speech 3.3-8b](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)
- [Speechmatics German medical model](https://www.speechmatics.com/company/articles-and-news/speechmatics-sets-new-standard-for-real-time-medical-transcription-with-german-and-nordic)
- [MultiMed multilingual medical ASR](https://huggingface.co/leduckhai/MultiMed)
- [Charité German medical anamnesis benchmark](https://arxiv.org/abs/2601.19945)
- [Modal domain-vocab Whisper tutorial](https://modal.com/docs/examples/fine_tune_asr)
- [Google MedASR](https://huggingface.co/google/medasr) — confirmed English-only, rejected
- [ml6 Dutch Whisper fine-tune sweep](https://blog.ml6.eu/fine-tuning-whisper-for-dutch-language-the-crucial-role-of-size-dd5a7012d45f)
- [HF Whisper fine-tuning blog](https://huggingface.co/blog/fine-tune-whisper)
- [HF audio course Ch 5](https://huggingface.co/learn/audio-course/en/chapter5/fine-tuning)
- [arXiv 2412.15726 Swiss-German Whisper large-v2 SOTA](https://arxiv.org/html/2412.15726v1)
- [Calm-Whisper hallucination on silence](https://arxiv.org/html/2505.12969v1)
- [openai/whisper #759 fine-tuning thread](https://github.com/openai/whisper/discussions/759)
- [openai/whisper #1118 30 s limit](https://github.com/openai/whisper/discussions/1118)
- [openai/whisper #1454 language tokens](https://github.com/openai/whisper/discussions/1454)
- [openai/whisper #589 suppress tokens](https://github.com/openai/whisper/discussions/589)
- [Bofeng Huang HF fine-tuning event recap](https://medium.com/@bofenghuang7/what-i-learned-from-whisper-fine-tuning-event-2a68dab1862)
- [vasistalodagala/whisper-finetune](https://github.com/vasistalodagala/whisper-finetune)
- [oliverguhr/fullstop-punctuation-multilang-large](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large) — German punctuation/casing restoration
- [HAI-DEF forum: MedASR brace tokens](https://discuss.ai.google.dev/t/medasr-clarification-needed-on-handling-of-brace-tokens-and-preprocessing-rules-for-fine-tuning-decoding/116107)
- [CH-DE emergency-medical ASR benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC12628192/)
- [AWS LoRA fine-tune blog](https://aws.amazon.com/blogs/machine-learning/fine-tune-whisper-models-on-amazon-sagemaker-with-lora/)
- [Greek medical Whisper fine-tune (arXiv 2509.23550)](https://arxiv.org/html/2509.23550v1)
- [ivrit.ai training Whisper guide](https://www.ivrit.ai/en/2025/02/13/training-whisper/)
