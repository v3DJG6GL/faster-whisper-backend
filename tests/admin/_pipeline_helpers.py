"""Helpers shared by the /settings/test-pipeline dry-run tests. Import as
``from tests.admin._pipeline_helpers import ...``."""


def dry_run(client, rules, sample):
    """Dry-run `rules` over `sample`; return the 200 response's JSON."""
    r = client.post("/settings/test-pipeline", json={"sample": sample, "rules": rules})
    assert r.status_code == 200, r.text
    return r.json()


def list_rule(entries, name="rl"):
    """One enabled regex-list rule, as the editor posts it."""
    return {"name": name, "type": "regex-list", "enabled": True, "entries": entries}


def list_step(client, entries, sample="aaaa"):
    """Dry-run one regex-list rule over `sample`; return its step."""
    return dry_run(client, [list_rule(entries)], sample)["steps"][0]


def upper_rule(pattern, name="up"):
    """One enabled callback:upper rule, as the editor posts it."""
    return {"name": name, "type": "callback:upper", "enabled": True, "pattern": pattern}
