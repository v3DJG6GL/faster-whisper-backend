"""Settings: the factory defaults and env layer (config), the admin schema and
override persistence (config_store), the env-var rename table
(config_renames) and the per-request resolver (effective_config). This file
must never import anything: it runs during config's own import.
"""
