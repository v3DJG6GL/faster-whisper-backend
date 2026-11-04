"""Helpers shared by the /settings/test-pipeline dry-run tests. Import as
``from tests.admin._pipeline_helpers import ...``."""


def list_step(client, entries, sample="aaaa"):
    """Dry-run one regex-list rule over `sample`; return its step."""
    r = client.post("/settings/test-pipeline", json={
        "sample": sample,
        "rules": [{"name": "rl", "type": "regex-list", "enabled": True,
                   "entries": entries}],
    })
    assert r.status_code == 200, r.text
    return r.json()["steps"][0]
