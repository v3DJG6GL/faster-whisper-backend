"""Shared infrastructure with no domain of its own: store helpers
(store_common, atomic_json), page chrome and template loading (web_common,
templates, home_routes), logging setup, the job registry, child-process and
language helpers, the SSRF address policy (net_policy, also loaded BY PATH by
the yt-dlp guard plugin). A module that serves one domain belongs in that
domain's package (pipeline/, transcription/, translation/, media/, ...), not
here. Keep this file import-free.
"""
