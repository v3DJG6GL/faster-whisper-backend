"""Pin that the CI checkouts never hand the run's token to a git credential
helper (same grep approach as test_ci_workflow_config.py)."""
from tests.conftest import read_repo_text as _read


def test_checkout_fetches_bypass_every_credential_helper():
    # git approves URL-embedded credentials into every configured helper
    # after an authenticated fetch; Git for Windows ships the system-wide
    # Credential Manager, so the host-mode VM would keep the token.
    ci = _read(".forgejo", "workflows", "ci.yml")
    fetches = [ln for ln in ci.splitlines()
               if not ln.strip().startswith("#") and " fetch " in ln
               and "url" in ln]
    assert len(fetches) == 6
    for ln in fetches:
        assert "git -c protocol.version=2 -c credential.helper= fetch " in ln, ln
