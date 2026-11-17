"""Pin the invariants of the CI/release workflows, renovate.json, the
ignore files and the Dockerfiles.

None of these has a runtime under test, so the tests grep the files for the
load-bearing lines (same approach as test_installer_scripts.py): a
regression that drops one of them silently brings back a guard that never
runs, an unbounded registry scan, or a release that can never be repaired.
"""
import json
import re

from tests.conftest import read_repo_text as _read

INSTANCE_HOST = "forgejo.informethic.ch"


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


def test_windows_leg_runs_in_its_own_venv():
    # The host-mode runner takes concurrent jobs: installing into the VM's
    # system python let a Renovate PR swap av 18 for 19 under a running main
    # job (run 1373).
    ci = _read(".forgejo", "workflows", "ci.yml")
    leg = ci[ci.index("  test-windows:"):]
    leg = leg[:leg.index("\n  all-ci:")]
    code = "\n".join(_code_lines(leg))
    assert "python -m venv .venv" in code
    assert code.index("python -m venv .venv") < code.index("-r requirements.txt")
    calls = re.findall(r"^\s*(\S+) -m (?:pip|pytest|uv)\b", code, re.M)
    assert calls and set(calls) == {r".venv\Scripts\python"}, calls
    # uv installs into the interpreter it is told to, not the one it runs
    # from: the requirements install must name the venv's.
    for line in re.findall(r"^.* -m uv pip install .*$", code, re.M):
        assert r"--python .venv\Scripts\python.exe" in line, line


def test_suite_filter_skips_the_suite_only_for_dockerfile_only_prs():
    # The deny-list's one skip arm: widening it lets a PR skip the suite for
    # files the tests read. Push/tag/dispatch runs, an undiffable PR and a
    # workflow change run every job.
    ci = _read(".forgejo", "workflows", "ci.yml")
    step = ci[ci.index("- name: Select the jobs"):ci.index("\n  test:")]
    assert re.findall(r"^\s*(.+)\) ;;$", step, re.M) == ["Dockerfile | Dockerfile.gpu"]
    assert 'if [ "$EVENT" != pull_request ]; then\n            every' in step
    assert '.forgejo/*) every "$f changed"; exit 0 ;;' in step
    assert 'every "no diff"' in step


def test_all_ci_rejects_a_failed_filter_and_unfiltered_windows_skips():
    # A failed `changes` job leaves its outputs empty; the gate must not
    # read that as "nothing to test", and Windows may only be skipped when
    # the filter said so.
    ci = _read(".forgejo", "workflows", "ci.yml")
    gate = ci[ci.index("  all-ci:"):ci.index("\n  build:")]
    assert "needs: [changes, test, test-windows]" in gate
    assert 'test "${{ needs.changes.result }}" = "success"' in gate
    assert ('if [ "${{ needs.changes.outputs.suite }}" = "false" ]; then\n'
            '              test "${{ needs.test-windows.result }}" = "skipped"') in gate


def test_test_legs_use_a_fixed_worker_count():
    # -n auto took all 80 CPUs of the shared runner host per leg and starved
    # the wall-clock tests (runs 1642-1646).
    ci = _read(".forgejo", "workflows", "ci.yml")
    runs = re.findall(r"pytest -n (\S+)", "\n".join(_code_lines(ci)))
    assert runs and all(n.isdigit() for n in runs), runs


def test_checkout_steps_derive_the_instance_host():
    # One literal for the whole file (env.REGISTRY); the fetch URLs read it
    # through GIT_HOST, so an instance rename is a one-line edit instead of
    # a fetch-time DNS/auth error on both test legs.
    ci = _read(".forgejo", "workflows", "ci.yml")
    carriers = [ln for ln in _code_lines(ci) if INSTANCE_HOST in ln]
    assert carriers == ["  REGISTRY: %s" % INSTANCE_HOST]
    assert ci.count("${GIT_TOKEN}@${GIT_HOST}/") == 1
    assert ci.count("${env:GIT_TOKEN}@${env:GIT_HOST}/") == 1


def test_both_checkouts_test_the_runs_own_commit():
    # all-ci's status and the release dispatch both name github.sha, but
    # main-push runs are serialized, so a queued run starts after later
    # pushes moved the ref: checking out the fetched tip tested a newer tree
    # and released a commit that never ran.
    ci = _read(".forgejo", "workflows", "ci.yml")
    for job, end in (("  test:", "\n  test-windows:"), ("  test-windows:", "\n  all-ci:")):
        leg = ci[ci.index(job):]
        step = leg[leg.index("- name: Checkout"):leg.index(end)]
        step = step[:step.index("\n      - name:")]
        code = "\n".join(_code_lines(step))
        assert "GIT_SHA: ${{ github.sha }}" in code
        assert "checkout -q --force FETCH_HEAD" not in code
        assert re.search(r'git checkout -q --force "\$(?:env:)?GIT_SHA"', code)


def test_both_checkouts_clean_untracked_state():
    # `checkout --force` resets tracked files only; the Windows workspace is
    # a persistent directory, so .coverage/__pycache__/test-created dirs
    # would otherwise leak into the next run.
    ci = _read(".forgejo", "workflows", "ci.yml")
    assert ci.count("git clean -qffdx") == 2


def test_build_legend_names_every_extra_the_dockerfile_installs():
    # The legend is where a CI reader learns what INCLUDE_EXTRAS=1 costs.
    # Derived from the INCLUDE_EXTRAS install line, so a fourth extra added
    # there without a legend word (and a map entry here) fails.
    ci = _read(".forgejo", "workflows", "ci.yml")
    dockerfile = _read("Dockerfile")
    legend = ci[ci.index("cpu-full"):ci.index("gpu-full — Dockerfile.gpu")]
    block = dockerfile[dockerfile.index('if [ "${INCLUDE_EXTRAS}" = "1" ]'):]
    block = block[:block.index("\n    fi")]
    installed = set(re.findall(r"-r requirements-(\w+)\.txt", block))
    words = {"diarize": "diarization", "bgm": "separation",
             "translate": "translation"}
    assert installed == set(words)
    for word in words.values():
        assert word in legend


def test_tag_runs_bake_the_pushed_tag_not_git_describe():
    # Two lightweight tags on one commit (manual minor after the auto patch)
    # make `git describe` ambiguous; a tag run must bake the ref it runs for.
    ci = _read(".forgejo", "workflows", "ci.yml")
    step = ci[ci.index("Compute build version"):]
    step = step[:step.index("- name:", 10)]
    assert '$GITHUB_REF_TYPE" = tag' in step
    assert "$GITHUB_REF_NAME" in step
    assert "git describe --tags --always" in step


def test_all_release_tag_runs_share_one_concurrency_group():
    # Per-ref groups let two v* tag runs push :latest in parallel, and the
    # older one finishing last moved :latest backwards.
    ci = _read(".forgejo", "workflows", "ci.yml")
    assert "startsWith(github.ref, 'refs/tags/v') && 'ci-release'" in ci


def test_only_the_newest_release_tag_moves_latest():
    # A workflow_dispatch on an OLD v* tag publishes like a tag push; with
    # only the startsWith(refs/tags/v) check it moved :latest (and, through
    # the mirror, ghcr's latest*) back to that release.
    ci = _read(".forgejo", "workflows", "ci.yml")
    build = ci[ci.index("\n  build:\n"):]
    build = build[:build.index("\n  mirror:\n")]
    meta = build.index("uses: docker/metadata-action@")
    newest = build.index("id: newest")
    assert newest < meta, "the newest step must run before the metadata step"
    step = build[build.rindex("- name:", 0, newest):meta]
    assert "git ls-remote --tags" in step
    assert "GITHUB_REF_NAME" in step
    # Only strict vX.Y.Z tags compete for "newest": an rc/vendor tag sorts
    # after every release under sort -V and no release would move :latest.
    assert "grep -E '^v[0-9]+\\.[0-9]+\\.[0-9]+$'" in step
    latest = [ln for ln in _code_lines(build[meta:])
              if "type=raw,value=latest" in ln]
    assert len(latest) == 1
    assert "startsWith(github.ref, 'refs/tags/v')" in latest[0]
    assert "steps.newest.outputs.is_newest == 'true'" in latest[0]


# --- .forgejo/workflows/mirror-ghcr.yml -------------------------------------

def test_mirror_skips_already_mirrored_sha_tags():
    # The growing history of sha-<short> tags must not cost two digest
    # round trips per tag on every dispatch and cron, so one already on the
    # destination is skipped. Not unconditionally: ci.yml re-pushes
    # sha-<short> for a second tag on the same commit or a dispatch rebuild,
    # so a sha tag whose revision a release copy just touched still takes
    # the digest compare.
    mirror = _read(".forgejo", "workflows", "mirror-ghcr.yml")
    assert 'dst_tags=$(crane ls "$DST_IMAGE" 2>/dev/null || true)' in mirror
    body = mirror[mirror.index("for tag in $tags; do"):]
    assert "sha-*)" in body
    assert 'grep -qxF "$tag"' in body
    # Membership reads a here-string: `printf | grep -q` under pipefail
    # returns 141 (SIGPIPE, read as "not found") once the list outgrows the
    # 64 KiB pipe buffer.
    assert not [ln for ln in _code_lines(mirror) if "| grep -q" in ln]
    assert 'grep -qxF "$tag" <<< "$dst_tags"' in body
    assert 'grep -qxF "$lt" <<< "$tags"' in mirror
    assert 'touched="$touched sha-${rev:0:7}${variant}"' in body
    assert '*" $tag "*) ;;' in body
    # latest*/v* are re-pushable and keep the full compare.
    assert 'src_digest=$(crane digest "$SRC_IMAGE:$tag")' in body
    # The pushed commit rides along with ci.yml's dispatch, so its sha tags
    # take the compare even when no latest* copy touched that revision.
    assert "DISPATCH_SHA: ${{ inputs.sha }}" in mirror
    assert "for lt in latest latest-gpu latest-full latest-gpu-full; do" in mirror
    ci = _read(".forgejo", "workflows", "ci.yml")
    assert '\\"inputs\\":{\\"sha\\":\\"${GITHUB_SHA}\\"}' in ci


def test_crane_renovate_marker_is_adjacent_to_its_version_key():
    # customManagers:githubActionsVersions allows only whitespace between the
    # `# renovate:` marker and the `_VERSION:` key — an intervening comment
    # line silently detaches crane from Renovate (no bump PR ever appears).
    lines = _read(".forgejo", "workflows", "mirror-ghcr.yml").splitlines()
    marker = "# renovate: datasource=github-releases depName=google/go-containerregistry"
    idx = [i for i, ln in enumerate(lines) if ln.strip() == marker]
    assert len(idx) == 1
    assert lines[idx[0] + 1].strip().startswith("CRANE_VERSION:")


# --- .forgejo/workflows/release.yml -----------------------------------------

def test_rerun_for_an_already_tagged_commit_can_repair_its_release():
    # Tag push and release POST are two writes: if the second fails, the tag
    # stands with no changelog entry. The idempotence branch must therefore
    # emit the existing tag rather than an empty one, or the re-dispatch
    # skips the release step forever and still reports success.
    rel = _read(".forgejo", "workflows", "release.yml")
    branch = rel[rel.index('if [ "$latest_commit" = "$target" ] && { [ -n "$TARGET_SHA" ] || [ "$BUMP" = patch ]; }; then'):]
    branch = branch[:branch.index("# Monotonicity")]
    assert 'echo "tag=$latest" >> "$GITHUB_OUTPUT"' in branch
    assert 'echo "prev=${prev:-v0.0.0}" >> "$GITHUB_OUTPUT"' in branch
    assert 'echo "tag=" ' not in branch


def test_rerun_of_a_superseded_tagged_commit_repairs_its_own_release():
    # Main moved on and minted a newer tag while this commit's release POST
    # failed: the monotonicity branch must hand THIS commit's own tag to the
    # release step instead of reporting "superseded" with an empty tag. The
    # target is peeled to a commit first, so a tag-object input still matches
    # `--points-at` and the ancestor test.
    rel = _read(".forgejo", "workflows", "release.yml")
    mono = rel[rel.index("# Monotonicity"):]
    mono = mono[:mono.index('case "$BUMP" in')]
    assert 'git tag --points-at "$target"' in mono
    assert 'echo "tag=$own" >> "$GITHUB_OUTPUT"' in mono
    assert 'target=$(git rev-parse "$target^{commit}")' in rel


def test_release_creation_probes_before_posting():
    # Re-entering the step for an existing release must be a clean no-op.
    rel = _read(".forgejo", "workflows", "release.yml")
    step = rel[rel.index("Create the Forgejo release"):]
    probe = step.index('"$GITHUB_API_URL/repos/$REPO/releases/tags/$TAG"')
    post = step.index('"$GITHUB_API_URL/repos/$REPO/releases"')
    assert probe < post
    assert "already exists" in step


def test_release_notes_do_not_travel_through_argv():
    # v0.1.236's ~1,290-commit log hit "jq: Argument list too long" and the
    # release POST got an empty body (422); the notes must go via a file.
    rel = _read(".forgejo", "workflows", "release.yml")
    step = rel[rel.index("Create the Forgejo release"):]
    assert "--arg body" not in step
    assert '--rawfile body "$notes"' in step


def test_manual_minor_bump_is_not_swallowed_by_idempotence():
    # The continuous flow patch-tags every green main commit, so a manual
    # minor/major on HEAD always meets the idempotence precondition; it must
    # fall through (and not be reported "superseded" — a commit is its own
    # ancestor for merge-base --is-ancestor).
    rel = _read(".forgejo", "workflows", "release.yml")
    assert ('[ "$latest_commit" = "$target" ] && { [ -n "$TARGET_SHA" ] '
            '|| [ "$BUMP" = patch ]; }') in rel
    assert ('[ "$latest_commit" != "$target" ] && git merge-base '
            '--is-ancestor "$target" "$latest_commit"') in rel
    assert 'if [ "$latest_commit" = "$target" ]; then' not in rel


def test_manual_same_boundary_rerun_repairs_instead_of_minting():
    # A re-run of a manual minor/major whose tag already sits on this commit
    # (a failed release POST, a second dispatch serialized behind the first)
    # must take the idempotence branch, not mint the next minor/major.
    # v1.3.0 + minor → repair; v1.2.4 + minor → v1.3.0; v2.0.0 + major →
    # repair; v1.3.0 + major → v2.0.0.
    rel = _read(".forgejo", "workflows", "release.yml")
    assert ('if [ "$BUMP" = minor ] && [ "$PA" = 0 ]; then '
            'same_boundary=1; fi') in rel
    assert ('if [ "$BUMP" = major ] && [ "$MI" = 0 ] && [ "$PA" = 0 ]; then '
            'same_boundary=1; fi') in rel
    assert ('if [ "$latest_commit" = "$target" ] && [ -n "$same_boundary" ]; '
            'then') in rel
    # The gate rewrites BUMP=patch before the idempotence branch tests it.
    gate = rel.index('[ -n "$same_boundary" ]; then')
    assert rel.index("BUMP=patch", gate) < rel.index(
        '{ [ -n "$TARGET_SHA" ] || [ "$BUMP" = patch ]; }; then', gate)


def test_test_legs_end_a_hang_long_before_the_runners_limit():
    # A hung test kept the Windows leg running for the runner's 3 h maximum on
    # every push and PR for a week, and printed no test name: each leg has a
    # job limit, and pytest ends a hung test's worker with its stacks.
    ci = _read(".forgejo", "workflows", "ci.yml")
    for job, nxt in (("\n  test:\n", "\n  test-windows:\n"), ("\n  test-windows:\n", "\n  all-ci:\n")):
        body = ci[ci.index(job):ci.index(nxt)]
        limit = re.search(r"^    timeout-minutes: (\d+)$", body, re.M)
        assert limit and 10 <= int(limit.group(1)) <= 60, job
    ini = _read("pytest.ini")
    timeout = re.search(r"^faulthandler_timeout = (\d+)$", ini, re.M)
    assert timeout and int(timeout.group(1)) < 30 * 60
    assert re.search(r"^faulthandler_exit_on_timeout = true$", ini, re.M)
    # The ini keys need pytest >= 9.0 (exit_on_timeout); an older pytest
    # would only warn about an unknown key and leave hangs unbounded.
    floor = re.search(r"^pytest>=(\d+)\.", _read("requirements-dev.txt"), re.M)
    assert floor and int(floor.group(1)) >= 9


# --- scripts/check_requirements_resolve.py -----------------------------------

def _resolve_check():
    import importlib.util
    import os

    from faster_whisper_backend.paths import REPO_ROOT
    spec = importlib.util.spec_from_file_location(
        "check_requirements_resolve",
        os.path.join(REPO_ROOT, "scripts", "check_requirements_resolve.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_INSTALLERS = ("install-service.ps1", "install-service.sh", "Dockerfile", "Dockerfile.gpu")


def test_linux_leg_checks_that_every_install_set_resolves():
    # The suites install requirements.txt + -dev only. Without this step a
    # Renovate bump of any other file auto-merges untried on Windows (cuBLAS
    # 12.9.3.13, Linux wheels only, broke install-service.ps1 -Gpu).
    ci = _read(".forgejo", "workflows", "ci.yml")
    linux = ci[ci.index("\n  test:\n"):ci.index("\n  test-windows:\n")]
    step = linux[linux.index("- name: Check every install set resolves"):]
    step = step[:step.index("\n      - name:")]
    assert "run: python scripts/check_requirements_resolve.py\n" in step
    assert "if: needs.changes.outputs.suite == 'true'" in step
    # After uv is installed, and before the long suite so a bad bump fails fast.
    assert linux.index('pip install --no-cache-dir "uv') < linux.index("- name: Check every install")
    assert linux.index("- name: Check every install") < linux.index("- name: Run tests")


def test_install_sets_cover_every_requirements_file_on_both_platforms():
    # A new requirements-*.txt (or one dropped from the table) would be bumped
    # by Renovate unchecked. A file counts for a platform when a set names it,
    # or names a file that `-r`-includes it.
    import os

    from faster_whisper_backend.paths import REPO_ROOT
    chk = _resolve_check()
    files = sorted(f for f in os.listdir(REPO_ROOT)
                   if f.startswith("requirements") and f.endswith(".txt"))
    assert "requirements-gpu.txt" in files and len(files) >= 8
    includes = {f: set(re.findall(r"^-r (\S+)", _read(f), re.M)) for f in files}
    for platform in ("windows", "linux"):
        covered = {f for s in chk.INSTALL_SETS if platform in s.platforms for f in s.files}
        while True:
            more = {inc for f in covered for inc in includes.get(f, ())} - covered
            if not more:
                break
            covered |= more
        assert covered == set(files), (platform, sorted(set(files) - covered))
    assert set(chk.PLATFORMS) == {"windows", "linux"}
    # Windows is the platform no CI leg installs these on: it must be x86-64.
    assert chk.PLATFORMS["windows"] == "x86_64-pc-windows-msvc"


def test_install_sets_mirror_the_installers_indexes_and_inline_packages():
    # The table is hand-kept: an index or a package an installer passes to pip
    # outside a requirements file must be resolved too, with that index.
    chk = _resolve_check()
    text = "\n".join(_read(name) for name in _INSTALLERS)
    indexes = set(re.findall(r"--extra-index-url (https://\S+)", text))
    assert indexes == {i for s in chk.INSTALL_SETS for i in s.indexes}
    assert len(indexes) == 4
    # (Not the `requires = ["setuptools>=61"]` of the ps1's diffq stub project.)
    inline = set(re.findall(r'(?<!\[)"([a-z][a-z0-9-]*(?:\[[a-z]+\]|[=<>]=[^"\s]+))"', text))
    probed = {name for names in chk.EXCLUDED.values() for name in names}
    # `"diffq-fixed>=0.2"` is the ps1's wheel probe: excluded, not resolved.
    inline = {spec for spec in inline if re.split(r"[\[=<>]", spec)[0] not in probed}
    assert inline == {spec for s in chk.INSTALL_SETS for spec in s.specs}
    assert inline == {"audio-separator[gpu]", "onnxruntime-gpu==1.26.*"}
    # The pinned ORT build is installed --no-deps, and resolved the same way.
    ort = [s for s in chk.INSTALL_SETS if s.specs == ("onnxruntime-gpu==1.26.*",)]
    assert len(ort) == 1 and ort[0].no_deps
    # Each exclusion stands for a workaround the installers really have.
    assert "diffq-fixed stub install" in _read("install-service.ps1")
    assert "audio-separator's diffq dep may compile from source" in _read("install-service.sh")


def test_install_sets_resolve_for_the_images_python_with_wheels_only(tmp_path):
    chk = _resolve_check()
    base = re.search(r"^FROM python:(\d+\.\d+)-slim", _read("Dockerfile"), re.M).group(1)
    assert chk.PYTHON == base
    full_gpu = next(s for s in chk.INSTALL_SETS if s.name == "full extras, gpu")
    args = chk.resolve_args(full_gpu, "windows", tmp_path)
    assert args[:2] == ["pip", "compile"]
    for flag, value in (("--python-version", base),
                        ("--python-platform", "x86_64-pc-windows-msvc"),
                        # A package that must be compiled on the box is not installable.
                        ("--only-binary", ":all:"),
                        ("--extra-index-url", "https://download.pytorch.org/whl/cu126")):
        assert args[args.index(flag) + 1] == value
    exclude = args[args.index("--excludes") + 1]
    with open(exclude, encoding="utf-8") as fh:
        assert fh.read() == "diffq-fixed\n"
    with open(args[-1], encoding="utf-8") as fh:
        assert fh.read() == "audio-separator[gpu]\n"
    assert sum(a.endswith(("requirements-diarize.txt", "requirements-bgm.txt")) for a in args) == 2
    assert "--no-deps" not in args


def test_resolve_check_fails_when_one_set_cannot_be_resolved(monkeypatch, capsys):
    import subprocess
    import sys
    chk = _resolve_check()
    asked = []

    def fake_run(cmd, **kwargs):
        asked.append(cmd)
        bad = "x86_64-pc-windows-msvc" in cmd and any(a.endswith("requirements-gpu.txt") for a in cmd)
        return subprocess.CompletedProcess(
            cmd, 1 if bad else 0, stdout=None,
            stderr="error: nvidia-cublas-cu12==12.9.3.13 has no wheels\n" if bad else "")

    monkeypatch.setattr(chk, "uv_command", lambda: ["uv"])
    monkeypatch.setattr(chk.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["check_requirements_resolve.py", "--jobs", "1"])
    assert chk.main() == 1
    out = capsys.readouterr().out
    assert len(asked) == sum(len(s.platforms) for s in chk.INSTALL_SETS)
    assert "FAIL  windows  base + gpu\n" in out and "ok    linux    base + gpu\n" in out
    assert "nvidia-cublas-cu12==12.9.3.13 has no wheels" in out
    # Every set green: exit 0.
    monkeypatch.setattr(chk.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout=None, stderr=""))
    assert chk.main() == 0


# --- renovate.json -----------------------------------------------------------

def test_renovate_excludes_only_versions_that_the_gpu_floor_does_not_need():
    # A version with no Windows wheel is kept out of Renovate's PRs by
    # allowedVersions; the floor in requirements-gpu.txt must stay below it,
    # or the file would again ask for the version Windows cannot install.
    rules = json.loads(_read("renovate.json"))["packageRules"]
    hits = [r for r in rules if "allowedVersions" in r]
    assert [r["matchPackageNames"] for r in hits] == [["nvidia-cublas-cu12"]]
    allowed = hits[0]["allowedVersions"]
    assert allowed.startswith("!/") and allowed.endswith("/")
    excluded = re.compile(allowed[2:-1])
    assert excluded.search("12.9.3.13") and not excluded.search("12.9.4.1")
    floor = re.search(r"^nvidia-cublas-cu12>=([\d.]+),<13$", _read("requirements-gpu.txt"), re.M)
    assert floor and not excluded.search(floor.group(1))
    assert tuple(map(int, floor.group(1).split("."))) < (12, 9, 3, 13)

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
    # automerge itself is a maintainer policy call; this pins only that the
    # major-update override is documented next to the flag.
    alerts = json.loads(_read("renovate.json"))["vulnerabilityAlerts"]
    assert alerts["automerge"] is True
    assert "major" in alerts["description"].lower()


# --- .gitignore / .dockerignore -----------------------------------------------

def test_ignore_files_do_not_swallow_package_dirs():
    """`captures/` is the legacy repo-root raw-WAV dir (the current default is
    {DATA_DIR}/captures). In .gitignore
    an unanchored `captures/` also matches faster_whisper_backend/captures/
    and tests/captures/ (git would refuse their __init__.py); in
    .dockerignore the dangerous form was `**/captures/` (a bare `captures/`
    is already root-relative there, and a leading slash is disregarded —
    kept for symmetry). Either would drop the package from the checkout or
    the image, and main imports captures.store at top level, so the app
    would not start."""
    for fname in (".gitignore", ".dockerignore"):
        lines = [ln.strip() for ln in _read(fname).splitlines()]
        assert "/captures/" in lines, f"{fname}: anchor the captures/ ignore to the root"
        for bad in ("captures/", "**/captures/"):
            assert bad not in lines, f"{fname}: {bad!r} also matches package dirs"


def test_ignore_files_anchor_every_root_only_dir():
    """Same trap as captures/ for the other root-only dirs: an unanchored
    `logs/`, `tmp/` or `ffmpeg/` in .gitignore matches a same-named package
    or test dir at any depth, and `**/logs/` in .dockerignore silently drops
    a faster_whisper_backend/logs/ package from every image (a real probe
    build showed it). `*.lock` would swallow a committed tool lockfile."""
    git = [ln.strip() for ln in _read(".gitignore").splitlines()]
    for d in ("logs", "tmp", "ffmpeg", "secrets"):
        assert f"/{d}/" in git, f".gitignore: anchor {d}/ to the root"
        assert f"{d}/" not in git and f"**/{d}/" not in git
    assert "*.lock" not in git
    docker = [ln.strip() for ln in _read(".dockerignore").splitlines()]
    for d in ("logs", "tmp", "ffmpeg", "captures", "secrets"):
        assert f"**/{d}/" not in docker, f".dockerignore: **/{d}/ matches package dirs"
    # ...while the root-only exclusion itself stays.
    assert "secrets/" in docker


def test_dockerignore_mirrors_the_time_ledger_report():
    """scripts/time_ledger.py writes /time-ledger.html at the repo root by
    default; .gitignore keeps it out of the public mirror, and a local
    `docker build .` (COPY . .) must not bake it into an image either."""
    git = [ln.strip() for ln in _read(".gitignore").splitlines()]
    docker = [ln.strip() for ln in _read(".dockerignore").splitlines()]
    assert "/time-ledger.html" in git
    assert "/time-ledger.html" in docker


def test_rotated_logs_under_an_in_repo_data_dir_are_ignored():
    """LOG_FILE defaults to {DATA_DIR}/logs/whisper.log with rotation; under
    an in-checkout DATA_DIR not named data/ (WHISPER_DATA_DIR=./state) the
    root-anchored /logs/ misses it and `*.log` matches only the live file,
    so the .1/.2 backups (the pipeline trace logs transcript text) were
    committable, and main is mirrored publicly. Unanchored, like `*.log`."""
    git = [ln.strip() for ln in _read(".gitignore").splitlines()]
    assert "*.log.*" in git, ".gitignore: rotated *.log.N backups not ignored"
    assert "**/*.log.*" in [ln.strip() for ln in _read(".dockerignore").splitlines()]


def test_dockerignore_keeps_an_in_repo_data_dir_out_of_the_image():
    """An in-repo DATA_DIR not named data/ (WHISPER_DATA_DIR=./state) keeps
    capture audio, the capture export and retained link media below it;
    a root-only `*.wav` (Docker's filepath.Match: * never crosses a /)
    let a local `docker build` bake them into a layer."""
    docker = [ln.strip() for ln in _read(".dockerignore").splitlines()]
    for pat in ("**/manifest.jsonl", "**/*.wav", "**/*.mp3", "**/*.m4a",
                "**/url_media/"):
        assert pat in docker, f".dockerignore: {pat} missing"
    for bare in ("manifest.jsonl", "*.wav", "*.mp3", "*.m4a"):
        assert bare not in docker, f".dockerignore: {bare} is root-only"


def test_gitignore_keeps_retained_url_media_out_of_git():
    """URL_MEDIA_DIR ({DATA_DIR}/url_media) holds retained link downloads
    and uploaded videos in any container format; under an in-checkout
    DATA_DIR not named data/ the *.wav/*.mp3/*.m4a globs miss .webm/.mp4,
    and main is mirrored publicly."""
    git = [ln.strip() for ln in _read(".gitignore").splitlines()]
    assert "url_media/" in git, ".gitignore: url_media/ not ignored"


# --- Dockerfiles -------------------------------------------------------------

def test_gpu_loader_path_matches_the_python_base_image():
    """Dockerfile.gpu puts the pip-installed nvidia/* dirs on LD_LIBRARY_PATH
    by their site-packages path, which embeds the base image's python3.X.
    A base-image minor bump (Renovate: python:3.14-slim -> 3.15-slim) that
    misses the path builds green and then loses cuBLAS/cuDNN at runtime, so
    this fails the bump PR until the path follows."""
    import re

    gpu = _read("Dockerfile.gpu")
    base = re.search(r"^FROM python:(\d+\.\d+)-slim", gpu, re.M)
    assert base, "Dockerfile.gpu: FROM python:X.Y-slim not found"
    env = re.search(r"^ENV LD_LIBRARY_PATH=(\S+)", gpu, re.M)
    assert env, "Dockerfile.gpu: ENV LD_LIBRARY_PATH not found"
    versions = set(re.findall(r"/python(\d+\.\d+)/", env.group(1)))
    assert versions == {base.group(1)}, (
        f"LD_LIBRARY_PATH names python{sorted(versions)}, base image is {base.group(1)}")
    # The CPU image must stay on the same base so both variants move together.
    cpu_from = re.search(r"^FROM (\S+)", _read("Dockerfile"), re.M).group(1)
    gpu_from = re.search(r"^FROM (\S+)", gpu, re.M).group(1)
    assert cpu_from == gpu_from


def test_images_disable_the_user_site_under_the_writable_home():
    """Both images run a non-root UID on the system interpreter with
    HOME=/tmp, so CPython's per-user site-packages resolves to a directory
    the runtime UID can create (/tmp/.local/lib/pythonX.Y/site-packages); a
    *.pth file there runs at the start of the server and of every child
    interpreter it spawns. PYTHONNOUSERSITE keeps it off the import path."""
    for name in ("Dockerfile", "Dockerfile.gpu"):
        runtime = _read(name).rpartition("\nUSER ")[2]
        assert "HOME=/tmp" in runtime, f"{name}: HOME=/tmp not set after USER"
        assert "PYTHONNOUSERSITE=1" in runtime, (
            f"{name}: PYTHONNOUSERSITE=1 missing beside the writable HOME")


def test_dockerignore_keeps_xdist_coverage_data_out_of_the_image():
    """pytest-xdist workers write `.coverage.<host>.<pid>.<rand>` beside
    `.coverage`; a killed run leaves them in the checkout, and `COPY . .`
    of a local build would bake the host name and source paths in."""
    import fnmatch

    for name in (".dockerignore", ".gitignore"):
        lines = [ln.strip() for ln in _read(name).splitlines()]
        assert ".coverage.*" in lines, f"{name}: .coverage.* missing"
    # The tracked coverage config must stay in the image's build context.
    assert not fnmatch.fnmatch(".coveragerc", ".coverage.*")


def test_python_base_image_minor_bumps_need_approval():
    rules = json.loads(_read("renovate.json"))["packageRules"]
    hits = [r for r in rules if r.get("matchPackageNames") == ["python"]
            and r.get("matchDatasources") == ["docker"]]
    assert len(hits) == 1
    rule = hits[0]
    assert "minor" in rule["matchUpdateTypes"]
    assert rule["automerge"] is False
    assert rule["dependencyDashboardApproval"] is True
