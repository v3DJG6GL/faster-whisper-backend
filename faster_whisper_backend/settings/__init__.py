"""Settings: the factory defaults and env layer (config), the admin schema and
the field-registry tables generated from it (schema), the per-field
descriptions (descriptions), override persistence (config_store), the
env-var / key / rule-slug rename table (config_renames), the per-request
resolver (effective_config) and the config version counter (version). This
file must never import anything: it runs during config's own import.
"""
