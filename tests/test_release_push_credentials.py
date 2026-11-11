"""Pin that the release job's tag push never hands the owner PAT to a git
credential helper (the ci.yml twin is test_ci_checkout_credentials.py)."""
from tests.conftest import read_repo_text as _read


def test_release_tag_push_bypasses_every_credential_helper():
    # git approves URL-embedded credentials into every configured helper
    # after it authenticates; an empty helper list keeps the PAT out of any
    # store a runner image (or a host-mode runner) brings along.
    release = _read(".forgejo", "workflows", "release.yml")
    pushes = [ln for ln in release.splitlines()
              if not ln.strip().startswith("#") and " push " in ln
              and "PUSH_TOKEN" in ln]
    assert len(pushes) == 1
    assert "git -c credential.helper= push " in pushes[0], pushes[0]
