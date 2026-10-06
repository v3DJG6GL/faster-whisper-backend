"""Text post-processing pipeline: the rules engine (engine), the dictation-map
compiler (dictation_map), the live-dictation seam hold-back (seam_holdback)
and the out-of-process regex safety probe (regex_guard). This file must never
import anything: settings/schema validators import regex_guard during
config's own import.
"""
