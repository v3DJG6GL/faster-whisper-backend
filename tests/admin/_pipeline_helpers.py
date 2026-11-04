"""Helpers shared by the /settings/test-pipeline dry-run tests. Import as
``from tests.admin._pipeline_helpers import ...``."""


def dry_run(client, rules, sample):
    """Dry-run `rules` over `sample`; return the 200 response's JSON."""
    r = client.post("/settings/test-pipeline", json={"sample": sample, "rules": rules})
    assert r.status_code == 200, r.text
    return r.json()


def list_step(client, entries, sample="aaaa"):
    """Dry-run one regex-list rule over `sample`; return its step."""
    return dry_run(client, [{"name": "rl", "type": "regex-list", "enabled": True,
                             "entries": entries}], sample)["steps"][0]
