"""Pin the CI/release workflow invariants fixed in the code review.

The workflows and renovate.json have no runtime under test, so these tests
grep the files for the load-bearing lines instead (same approach as
test_installer_scripts.py): a regression that drops one of them silently
brings back a guard that never runs, an unbounded registry scan, or a
release that can never be repaired. The CI itself cannot be executed here.
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INSTANCE_HOST = "forgejo.informethic.ch"


def _read(*rel):
    with open(os.path.join(REPO, *rel), encoding="utf-8") as fh:
        return fh.read()


def _code_lines(text):
    """Lines that are not pure comments — the parts a runner actually reads."""
    return [ln for ln in text.splitlines() if not ln.strip().startswith("#")]


# --- .forgejo/workflows/ci.yml ----------------------------------------------

def test_linux_test_leg_installs_node_and_ffmpeg():
    # python:<ver>-trixie ships neither, and their absence does not fail the
    # suite: test_inline_scripts_parse and the WebM/Opus transport tests skip
    # themselves, so the leg would report green having checked nothing.
    ci = _read(".forgejo", "workflows", "ci.yml")
    assert "apt-get install -y --no-install-recommends nodejs ffmpeg" in ci
    # Before the deps install, so a broken tool install fails early.
    assert ci.index("nodejs ffmpeg") < ci.index("pip install -r requirements.txt")


def test_windows_leg_fails_when_node_or_ffmpeg_is_missing():
    # The VM is hand-provisioned, so the parity leg must assert its tools
    # rather than skip the guards that need them.
    ci = _read(".forgejo", "workflows", "ci.yml")
    step = ci[ci.index("- name: Verify system test tools"):]
    step = step[:step.index("\n      - name:")]
    assert "@('node', 'ffmpeg')" in step
    assert "::error::" in step and "exit 1" in step


def test_checkout_steps_derive_the_instance_host():
    # One literal for the whole file (env.REGISTRY); the fetch URLs read it
    # through GIT_HOST, so an instance rename is a one-line edit instead of
    # a fetch-time DNS/auth error on both test legs.
    ci = _read(".forgejo", "workflows", "ci.yml")
    carriers = [ln for ln in _code_lines(ci) if INSTANCE_HOST in ln]
    assert carriers == ["  REGISTRY: %s" % INSTANCE_HOST]
    assert ci.count("${GIT_TOKEN}@${GIT_HOST}/") == 1
    assert ci.count("${env:GIT_TOKEN}@${env:GIT_HOST}/") == 1


def test_both_checkouts_clean_untracked_state():
    # `checkout --force` resets tracked files only; the Windows workspace is
    # a persistent directory, so .coverage/__pycache__/test-created dirs
    # would otherwise leak into the next run.
    ci = _read(".forgejo", "workflows", "ci.yml")
    assert ci.count("git clean -qffdx") == 2


def test_build_legend_names_every_extra_the_dockerfile_installs():
    # The legend is where a CI reader learns what INCLUDE_EXTRAS=1 costs.
    ci = _read(".forgejo", "workflows", "ci.yml")
    dockerfile = _read("Dockerfile")
    legend = ci[ci.index("cpu-full"):ci.index("gpu-full — Dockerfile.gpu")]
    for req in ("diarize", "bgm", "translate"):
        assert "requirements-%s.txt" % req in dockerfile
    assert "diarization" in legend
    assert "separation" in legend
    assert "translation" in legend


# --- .forgejo/workflows/mirror-ghcr.yml -------------------------------------

def test_mirror_skips_already_mirrored_sha_tags():
    # sha-<short> is content-addressed: presence on the destination is
    # identity, so the growing history of them must not cost two digest
    # round trips per tag on every dispatch and cron.
    mirror = _read(".forgejo", "workflows", "mirror-ghcr.yml")
    assert 'dst_tags=$(crane ls "$DST_IMAGE" 2>/dev/null || true)' in mirror
    body = mirror[mirror.index("for tag in $tags; do"):]
    assert "sha-*)" in body
    assert 'grep -qxF "$tag"' in body
    # latest*/v* are re-pushable and keep the full compare.
    assert 'src_digest=$(crane digest "$SRC_IMAGE:$tag")' in body


# --- .forgejo/workflows/release.yml -----------------------------------------

def test_rerun_for_an_already_tagged_commit_can_repair_its_release():
    # Tag push and release POST are two writes: if the second fails, the tag
    # stands with no changelog entry. The idempotence branch must therefore
    # emit the existing tag rather than an empty one, or the re-dispatch
    # skips the release step forever and still reports success.
    rel = _read(".forgejo", "workflows", "release.yml")
    branch = rel[rel.index('if [ "$latest_commit" = "$target" ]; then'):]
    branch = branch[:branch.index("# Monotonicity")]
    assert 'echo "tag=$latest" >> "$GITHUB_OUTPUT"' in branch
    assert 'echo "prev=${prev:-v0.0.0}" >> "$GITHUB_OUTPUT"' in branch
    assert 'echo "tag=" ' not in branch


def test_release_creation_probes_before_posting():
    # Re-entering the step for an existing release must be a clean no-op.
    rel = _read(".forgejo", "workflows", "release.yml")
    step = rel[rel.index("Create the Forgejo release"):]
    probe = step.index('"$GITHUB_API_URL/repos/$REPO/releases/tags/$TAG"')
    post = step.index('"$GITHUB_API_URL/repos/$REPO/releases"')
    assert probe < post
    assert "already exists" in step


# --- renovate.json -----------------------------------------------------------

def test_ytdlp_fast_lane_outranks_the_blanket_major_rule():
    # yt-dlp is date-versioned, so the year rollover is a pep440 *major*.
    # Renovate merges matching rules in array order with later keys winning,
    # so the fast lane only survives the rollover from the last position.
    rules = json.loads(_read("renovate.json"))["packageRules"]
    major = [i for i, r in enumerate(rules)
             if r.get("matchUpdateTypes") == ["major"] and "matchPackageNames" not in r]
    ytdlp = [i for i, r in enumerate(rules) if r.get("matchPackageNames") == ["yt-dlp"]]
    assert len(major) == 1 and len(ytdlp) == 1
    assert ytdlp[0] > major[0]
    assert ytdlp[0] == len(rules) - 1
    rule = rules[ytdlp[0]]
    assert rule["automerge"] is True
    # Explicit: last-wins is per property, so an omitted key would leave the
    # major rule's dashboard gate in force.
    assert rule["dependencyDashboardApproval"] is False


def test_vulnerability_alert_automerge_records_the_major_override():
    # The value itself is a maintainer policy call and is left as-is; what
    # the review requires is that the override is written down where the
    # flag lives.
    alerts = json.loads(_read("renovate.json"))["vulnerabilityAlerts"]
    assert alerts["automerge"] is True
    assert "major" in alerts["description"].lower()


# --- .gitignore / .dockerignore -----------------------------------------------

def test_ignore_files_do_not_swallow_package_dirs():
    """`captures/` is the Windows raw-WAV dir at the repo ROOT. Unanchored,
    the same pattern would also drop faster_whisper_backend/captures/ and
    tests/captures/ from git and from the Docker build context — silently,
    because main wraps the router import in try/except."""
    for fname in (".gitignore", ".dockerignore"):
        lines = [ln.strip() for ln in _read(fname).splitlines()]
        assert "/captures/" in lines, f"{fname}: anchor the captures/ ignore to the root"
        for bad in ("captures/", "**/captures/"):
            assert bad not in lines, f"{fname}: {bad!r} also matches package dirs"
