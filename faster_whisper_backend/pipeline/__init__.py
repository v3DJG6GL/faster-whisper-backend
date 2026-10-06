"""Text post-processing pipeline: the rules engine (engine), the dictation-map
compiler (dictation_map), the live-dictation seam hold-back (seam_holdback),
the out-of-process regex safety probe (regex_guard) and the config hot-apply
plus the shared PIPELINE_RULES lock (apply). This file must never
import anything: settings/schema validators import regex_guard during
config's own import.
"""
