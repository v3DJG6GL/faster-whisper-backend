"""model_sizes: the persisted measured-size ledger and the fit check.

Everything runs against a real temp file — the atomic-write/lock path is the
part most likely to break, so stubbing it out would test nothing.
"""
import json
import os

import pytest

from faster_whisper_backend.audio import bgm_separation
from faster_whisper_backend.core import atomic_json
from faster_whisper_backend.runtime import model_sizes
from faster_whisper_backend.settings import config as cfg
from tests.conftest import record_calls, write_bytes


GB = 1024 ** 3


def _symlink_or_skip(src, dst):
    try:
        os.symlink(src, dst)
    except OSError:
        pytest.skip("no symlink support")


def _hf_repo(tmp_path, monkeypatch, repo):
    """Point HF_HOME at tmp_path/hf; return its hub/models--repo dir (not created)."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    return tmp_path / "hf" / "hub" / f"models--{repo}"


def _bytes_src(name, device, compute):
    got = model_sizes.lookup(name, device, compute)
    return got["bytes"], got["src"]


def _gpu_free(monkeypatch, n):
    monkeypatch.setattr(model_sizes.system_stats, "gpu_mem_free_bytes",
                        lambda *a: n, raising=False)


@pytest.fixture
def ledger(model_sizes_ledger):
    """The conftest model_sizes_ledger (PATH + the def-time path defaults)."""
    return model_sizes_ledger


def test_record_estimate_roundtrip_survives_a_restart(ledger):
    model_sizes.record("large-v3", "cuda", "float16", 3 * GB)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 3 * GB
    assert os.path.exists(ledger)

    # Simulate a fresh process: the module-level cache is empty, the file isn't.
    model_sizes._reset_for_tests()
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 3 * GB


def test_smaller_resample_never_lowers_the_stored_size(ledger):
    model_sizes.record("large-v3", "cuda", "float16", 4 * GB)
    # CTranslate2's caching allocator makes a re-load under-report; the ledger
    # must keep the high-water mark or the fit check turns into an OOM.
    model_sizes.record("large-v3", "cuda", "float16", 1 * GB)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 4 * GB

    # Below the high-water mark is no news: the file is not rewritten.
    doc = json.loads(open(ledger, encoding="utf-8").read())
    assert doc["models"]["large-v3|cuda|float16"]["n"] == 1


def test_placement_is_part_of_the_key(ledger):
    model_sizes.record("large-v3", "cuda", "float16", 3 * GB)
    model_sizes.record("large-v3", "cpu", "int8", 1 * GB)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 3 * GB
    assert model_sizes.estimate("large-v3", "cpu", "int8") == 1 * GB


def test_any_device_fallback_then_none(ledger, monkeypatch):
    # A cuda measurement stands in for another placement as a proxy (a cpu
    # row would be disk-grade and never take this path). No disk walk, so
    # the real hub cache cannot decide the outcome.
    monkeypatch.setattr(model_sizes, "disk_size", lambda name: None)
    model_sizes.record("gguf:some/repo", "cuda", "int8_float16", 2 * GB)
    assert _bytes_src("gguf:some/repo", "cpu", "int8") == (2 * GB, "proxy")
    assert model_sizes.estimate("gguf:some/repo", "cpu", "int8") == 2 * GB
    assert model_sizes.estimate("never-loaded", "cuda", "float16") is None


def test_one_percent_drift_does_not_rewrite_the_file(ledger, monkeypatch):
    model_sizes.record("large-v3", "cuda", "float16", 1000 * 1000 * 1000)

    writes = record_calls(monkeypatch, model_sizes, "_write_locked", lambda *a, **k: 1)

    model_sizes.record("large-v3", "cuda", "float16", 1010 * 1000 * 1000)  # +1%
    assert writes == []
    model_sizes.record("large-v3", "cuda", "float16", 1200 * 1000 * 1000)  # +20%
    assert len(writes) == 1
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 1200 * 1000 * 1000


def test_fits_cuda_verdicts_and_the_reserve_boundary(ledger, monkeypatch):
    model_sizes.record("large-v3", "cuda", "float16", 3 * GB)
    reserve = 1 * GB

    _gpu_free(monkeypatch, 8 * GB)
    assert model_sizes.fits("large-v3", "cuda", "float16",
                            reserve_bytes=reserve) == (True, None)
    # free - need == reserve is still a fit; one byte less is not.
    _gpu_free(monkeypatch, 4 * GB)
    assert model_sizes.fits("large-v3", "cuda", "float16",
                            reserve_bytes=reserve) == (True, None)
    _gpu_free(monkeypatch, 4 * GB - 1)
    assert model_sizes.fits("large-v3", "cuda", "float16",
                            reserve_bytes=reserve) == (False, "insufficient_vram")


def test_fits_reports_unknowns_apart(ledger, monkeypatch):
    _gpu_free(monkeypatch, 8 * GB)
    # Never measured: "cannot say", NOT a refusal.
    assert model_sizes.fits("never-loaded", "cuda", "float16",
                            reserve_bytes=0) == (None, "size_unknown")

    # NVML absent: we refuse rather than guess on the GPU...
    _gpu_free(monkeypatch, None)
    model_sizes.record("large-v3", "cuda", "float16", 3 * GB)
    assert model_sizes.fits("large-v3", "cuda", "float16",
                            reserve_bytes=0) == (False, "vram_unknown")


def test_fits_cpu_uses_psutil_even_without_nvml(ledger, monkeypatch):
    _gpu_free(monkeypatch, None)
    model_sizes.record("large-v3", "cpu", "int8", 2 * GB)

    class _VM:
        available = 5 * GB
    monkeypatch.setattr(model_sizes.psutil, "virtual_memory", lambda: _VM())

    assert model_sizes.fits("large-v3", "cpu", "int8",
                            reserve_bytes=3 * GB) == (True, None)
    assert model_sizes.fits("large-v3", "cpu", "int8",
                            reserve_bytes=3 * GB + 1) == (False, "insufficient_ram")


@pytest.mark.parametrize("body", [
    "{not json at all",
    '{"version": 1, "models": {"a|cuda|f16": {"byt',   # truncated write
    '{"version": 99, "models": {"a|cuda|f16": {"bytes": 1}}}',
    '["not", "an", "object"]',
])
def test_unreadable_file_degrades_to_no_data(ledger, body):
    with open(ledger, "w", encoding="utf-8") as f:
        f.write(body)
    model_sizes._reset_for_tests()
    assert model_sizes.estimate("a", "cuda", "f16") is None


@pytest.mark.parametrize("row", [
    '{"bytes": Infinity}',
    '{"bytes": NaN}',
    '{"bytes": -5}',
    '{"bytes": true}',
])
def test_a_nonsense_size_row_is_dropped(ledger, monkeypatch, row):
    """json.load accepts Infinity / NaN and a bool is an int: lookup()'s
    int() would raise (and 500 /stats), and a negative size would make
    fits() always say yes."""
    monkeypatch.setattr(model_sizes, "disk_size", lambda name: None)
    with open(ledger, "w", encoding="utf-8") as f:
        f.write('{"version": 1, "models": {"a|cuda|f16": %s}}' % row)
    model_sizes._reset_for_tests()
    assert model_sizes.lookup("a", "cuda", "f16") is None
    assert model_sizes.lookup("a", "cpu", "int8") is None
    assert model_sizes.fits("a", "cpu", "int8",
                            reserve_bytes=0) == (None, "size_unknown")


# Explicit ids: the default one is the body, and pytest puts it in
# PYTEST_CURRENT_TEST — Windows refuses an env value over 32767 characters.
@pytest.mark.parametrize("body", [
    '{"version": 1, "models": {"a|cuda|f16": {"bytes": 1%s}}}' % ("0" * 400),
    '{"version": 1, "models": {"a|cuda|f16": {"bytes": 10, "n": 1%s}}}'
    % ("0" * 400),
    "[" * 100000 + "]" * 100000,
], ids=["huge-bytes", "huge-count", "deep-nesting"])
def test_an_overflowing_or_deeply_nested_file_degrades_to_no_data(
        ledger, monkeypatch, body):
    """_read NEVER raises: math.isfinite() cannot convert a 400-digit JSON
    int, and json.load recurses on a deeply nested document. Either one
    escaping would 500 /stats and break a preload fit check."""
    monkeypatch.setattr(model_sizes, "disk_size", lambda name: None)
    with open(ledger, "w", encoding="utf-8") as f:
        f.write(body)
    model_sizes._reset_for_tests()
    assert model_sizes._read(ledger) == {}
    assert model_sizes.lookup("a", "cuda", "f16") is None


def test_a_nonsense_count_reads_as_zero(ledger):
    with open(ledger, "w", encoding="utf-8") as f:
        f.write('{"version": 1, "models": {"a|cuda|f16": '
                '{"bytes": 10, "n": "many"}}}')
    model_sizes._reset_for_tests()
    assert model_sizes.lookup("a", "cuda", "f16")["n"] == 0
    assert model_sizes.lookup("a", "cpu", "int8")["n"] == 0


# Disk-size prior: without it a never-loaded model cannot be sized, so
# preload refuses it and it is never measured. On a fresh install that
# deadlock covers every model, silently.

def test_disk_size_reads_a_uvr_onnx_file(ledger, tmp_path, monkeypatch):
    root = tmp_path / "dl"
    write_bytes(root / "audio-separator" / "UVR-MDX-NET-Inst_HQ_4.onnx", b"x" * 4096)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", str(root), raising=False)

    assert model_sizes.disk_size("uvr:UVR-MDX-NET-Inst_HQ_4") == 4096
    # The `.onnx` suffix is implied by the friendly name, as elsewhere.
    assert model_sizes.disk_size("uvr:UVR-MDX-NET-Inst_HQ_4.onnx") == 4096


def test_disk_size_sums_a_hf_repo_dir(ledger, tmp_path, monkeypatch):
    d = _hf_repo(tmp_path, monkeypatch,
                 "pyannote--speaker-diarization-3.1") / "blobs"
    write_bytes(d / "a", b"x" * 1000)
    write_bytes(d / "b", b"x" * 2000)

    assert model_sizes.disk_size("pyannote:pyannote/speaker-diarization-3.1") == 3000


def test_disk_size_counts_snapshot_symlinks_once(ledger, tmp_path, monkeypatch):
    """A hub cache's snapshots/<rev>/* are symlinks into blobs/; following
    them as well would double every weight."""
    repo = _hf_repo(tmp_path, monkeypatch, "org--repo")
    (repo / "snapshots" / "rev").mkdir(parents=True)
    write_bytes(repo / "blobs" / "a", b"x" * 1000)
    _symlink_or_skip(os.path.join("..", "..", "blobs", "a"),
                     repo / "snapshots" / "rev" / "a")

    assert model_sizes.disk_size("pyannote:org/repo") == 1000


def test_disk_size_counts_a_symlink_that_points_out_of_the_tree(ledger,
                                                                tmp_path):
    """A local CT2 dir (DEFAULT_MODEL, sized in place) whose model.bin links
    to another disk: the weights live outside the walked tree, so skipping
    links would size only config.json."""
    model = tmp_path / "model"
    elsewhere = tmp_path / "elsewhere"
    write_bytes(elsewhere / "model.bin", b"x" * 100_000)
    write_bytes(model / "config.json", b"{}  ")
    _symlink_or_skip(elsewhere / "model.bin", model / "model.bin")
    assert model_sizes.disk_size(str(model)) == 100_004


def test_disk_size_sizes_an_all_symlink_dir(ledger, tmp_path):
    """A hub snapshots/<rev> dir used directly as the model path holds only
    links into ../../blobs: it has a size, not None (size_unknown)."""
    blobs = tmp_path / "blobs"
    rev = tmp_path / "snapshots" / "rev"
    rev.mkdir(parents=True)
    write_bytes(blobs / "a", b"x" * 1000)
    write_bytes(blobs / "b", b"x" * 24)
    _symlink_or_skip(blobs / "a", rev / "model.bin")
    _symlink_or_skip(blobs / "b", rev / "config.json")
    _symlink_or_skip(tmp_path / "gone", rev / "dangling")   # skipped, not fatal
    assert model_sizes.disk_size(str(rev)) == 1024


def test_disk_size_sizes_only_the_requested_gguf_quant(ledger, tmp_path,
                                                       monkeypatch):
    """An operator who ever fetched a second quant of the same repo must
    not be charged for both files when sizing either one."""
    snap = _hf_repo(tmp_path, monkeypatch,
                    "tencent--HY-MT1.5-7B-GGUF") / "snapshots" / "rev"
    write_bytes(snap / "HY-MT1.5-7B-Q4_K_M.gguf", b"x" * 1000)
    write_bytes(snap / "HY-MT1.5-7B-Q8_0.gguf", b"x" * 2000)

    assert model_sizes.disk_size("gguf:tencent/HY-MT1.5-7B-GGUF:Q4_K_M") == 1000
    assert model_sizes.disk_size("gguf:tencent/HY-MT1.5-7B-GGUF:q8_0") == 2000
    # No single match: unknown, not a guess.
    assert model_sizes.disk_size("gguf:tencent/HY-MT1.5-7B-GGUF:Q5_K_M") is None
    # A quant-less ref resolves with "*.gguf" (translation.engine): two
    # quants are no single match, never the sum of the repo dir.
    assert model_sizes.disk_size("gguf:tencent/HY-MT1.5-7B-GGUF") is None
    (snap / "HY-MT1.5-7B-Q8_0.gguf").unlink()
    assert model_sizes.disk_size("gguf:tencent/HY-MT1.5-7B-GGUF") == 1000


def test_disk_size_finds_a_gguf_quant_cached_in_an_older_snapshot(
        ledger, tmp_path, monkeypatch):
    """translation.engine resolves a quant across every cached revision, so
    a quant fetched before the repo moved on must keep its disk prior."""
    snaps = _hf_repo(tmp_path, monkeypatch, "org--x") / "snapshots"
    write_bytes(snaps / "A" / "x-Q4_K_M.gguf", b"x" * 1000)
    write_bytes(snaps / "B" / "x-Q8_0.gguf", b"x" * 2000)
    # Dated by their newest file, like scan_cache_dir dates a revision.
    os.utime(snaps / "A" / "x-Q4_K_M.gguf", (1_000_000, 1_000_000))
    os.utime(snaps / "B" / "x-Q8_0.gguf", (2_000_000, 2_000_000))

    assert model_sizes.disk_size("gguf:org/x:Q4_K_M") == 1000
    assert model_sizes.disk_size("gguf:org/x:Q8_0") == 2000
    # Quant-less: the newest revision holding any .gguf decides.
    assert model_sizes.disk_size("gguf:org/x") == 2000
    # The same name in two revisions: the newest copy is the one loaded.
    (snaps / "B" / "x-Q4_K_M.gguf").write_bytes(b"x" * 1500)
    os.utime(snaps / "B" / "x-Q4_K_M.gguf", (2_000_000, 2_000_000))
    assert model_sizes.disk_size("gguf:org/x:Q4_K_M") == 1500
    # Two quants in the deciding revision: no single match.
    assert model_sizes.disk_size("gguf:org/x") is None


def test_disk_size_sizes_the_gguf_the_loader_opens_after_a_rename(
        ledger, tmp_path, monkeypatch):
    """engine._cached_gguf lets the revision refs/main points at decide,
    then the newest: a file renamed upstream (old-Q4_K_M in an old
    revision, new-Q4_K_M in main) loads fine and must keep its disk prior
    rather than read as "no single match" -> fits() size_unknown forever."""
    repo = _hf_repo(tmp_path, monkeypatch, "org--x")
    blobs = repo / "blobs"
    for rev, name, size, t in (("a" * 40, "old-Q4_K_M.gguf", 1000, 1_000_000),
                               ("b" * 40, "new-Q4_K_M.gguf", 2000, 2_000_000)):
        blob = write_bytes(blobs / rev[:8], b"x" * size)
        snap = repo / "snapshots" / rev
        snap.mkdir(parents=True)
        _symlink_or_skip(os.path.join("..", "..", "blobs", rev[:8]), snap / name)
        os.utime(blob, (t, t))
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("b" * 40)

    assert model_sizes.disk_size("gguf:org/x:Q4_K_M") == 2000
    assert model_sizes.disk_size("gguf:org/x") == 2000
    # main beats a newer revision: the loader opens main's copy.
    os.utime(blobs / ("a" * 8), (3_000_000, 3_000_000))
    assert model_sizes.disk_size("gguf:org/x:Q4_K_M") == 2000
    (repo / "refs" / "main").write_text("a" * 40)
    assert model_sizes.disk_size("gguf:org/x:Q4_K_M") == 1000


def test_disk_size_is_none_when_nothing_is_there(ledger, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "nope"))
    assert model_sizes.disk_size("gguf:no/such") is None


def test_estimate_falls_back_to_disk_then_prefers_a_measurement(
        ledger, tmp_path, monkeypatch):
    d = _hf_repo(tmp_path, monkeypatch, "openai--whisper-tiny")
    write_bytes(d / "model.bin", b"x" * 8192)
    # The whisper branch checks <DOWNLOAD_ROOT> first; the real default
    # ("/models") may hold a real whisper-tiny on a deployment box.
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", "", raising=False)

    # Never measured → the prior stands in, and fits() can now decide.
    assert model_sizes.estimate("openai/whisper-tiny", "cuda", "float16") == 8192
    assert model_sizes.fits("openai/whisper-tiny", "cpu", "float16",
                            reserve_bytes=0)[1] != "size_unknown"

    # A real measurement supersedes it.
    model_sizes.record("openai/whisper-tiny", "cuda", "float16", 99 * 1024)
    assert model_sizes.estimate("openai/whisper-tiny", "cuda",
                                "float16") == 99 * 1024


def test_disk_size_uvr_falls_back_to_tempdir_without_download_root(
        ledger, tmp_path, monkeypatch):
    """bgm_separation loads from <tempdir>/audio-separator when DOWNLOAD_ROOT
    is unset — the sizing must look in the same place, not give up."""
    write_bytes(tmp_path / "audio-separator" / "Foo.onnx", b"x" * 2048)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", "", raising=False)
    monkeypatch.setattr(model_sizes.tempfile, "gettempdir",
                        lambda: str(tmp_path))

    assert model_sizes.disk_size("uvr:Foo") == 2048


def test_uvr_path_matches_the_separator_model_dir(tmp_path, monkeypatch):
    """model_sizes mirrors bgm_separation.model_file_dir() instead of
    importing it; this keeps the two from drifting apart."""
    monkeypatch.setattr(model_sizes.tempfile, "gettempdir",
                        lambda: str(tmp_path))
    for root in (str(tmp_path / "dl"), ""):
        monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", root, raising=False)
        assert os.path.dirname(model_sizes._model_path("uvr:Foo")) \
            == bgm_separation.model_file_dir()


def test_uvr_dir_expands_a_tilde_download_root(tmp_path, monkeypatch):
    """A .env DOWNLOAD_ROOT=~/models reaches cfg unexpanded; Separator
    makedirs model_file_dir as given, so the UVR models must land under the
    home dir like the whisper/HF ones, not in a literal '<cwd>/~' tree."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # expanduser on Windows
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", "~/models", raising=False)
    want = str(tmp_path / "models" / "audio-separator")
    norm = os.path.normpath  # "~/models" keeps its "/" on Windows

    assert norm(bgm_separation.model_file_dir()) == want
    assert norm(model_sizes._model_path("uvr:Foo")) \
        == os.path.join(want, "Foo.onnx")


def test_measurement_replaces_a_larger_disk_prior(ledger):
    """The disk walk over-counts (every revision, fp32 blobs); the first REAL
    measurement must win even when it is smaller, or the prior is unbeatable
    and fits() refuses a model that would load fine."""
    model_sizes.record("large-v3", "cuda", "float16", 6 * GB, measured=False)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 6 * GB

    model_sizes.record("large-v3", "cuda", "float16", 3 * GB)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 3 * GB
    doc = json.loads(open(ledger, encoding="utf-8").read())
    assert doc["models"]["large-v3|cuda|float16"]["src"] == "measured"

    # ...and a later disk prior never drags a measured row anywhere.
    model_sizes.record("large-v3", "cuda", "float16", 9 * GB, measured=False)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 3 * GB
    # Measured-vs-measured still keeps the high-water mark.
    model_sizes.record("large-v3", "cuda", "float16", 1 * GB)
    assert model_sizes.estimate("large-v3", "cuda", "float16") == 3 * GB


def test_disk_size_defaults_to_the_hub_cache_without_any_root(
        ledger, tmp_path, monkeypatch):
    """Empty DOWNLOAD_ROOT + no HF_HOME is a documented install
    (.env.example: 'empty = standard HF cache ~/.cache/huggingface'); the
    prior must look where the hub itself writes (its HF_HUB_CACHE, frozen at
    import) instead of giving up."""
    from huggingface_hub import constants
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", "", raising=False)
    hub = tmp_path / ".cache" / "huggingface" / "hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub))
    d = hub / "models--org--repo" / "snapshots" / "rev"
    write_bytes(d / "x.gguf", b"x" * 1234)

    assert model_sizes.disk_size("gguf:org/repo") == 1234


def test_disk_size_finds_whisper_under_download_root(
        ledger, tmp_path, monkeypatch):
    """transcription.models passes DOWNLOAD_ROOT itself as snapshot_download's
    cache_dir and maps 'large-v3' through faster_whisper's _MODELS table — the
    prior must look where the bytes actually land."""
    pytest.importorskip("faster_whisper")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", str(tmp_path), raising=False)
    d = tmp_path / "models--Systran--faster-whisper-large-v3"
    write_bytes(d / "model.bin", b"x" * 5000)

    assert model_sizes.disk_size("large-v3") == 5000
    assert model_sizes.disk_size("Systran/faster-whisper-large-v3") == 5000


def test_record_merges_a_peer_workers_row_instead_of_clobbering_it(ledger):
    """Two workers share the file. A peer's row written while this process
    holds a stale in-memory copy (same mtime — writes within one clock
    tick) must survive our next record(), which rewrites the whole
    document."""
    model_sizes.record("a", "cuda", "float16", 1 * GB)      # warms the cache
    doc = json.loads(open(ledger, encoding="utf-8").read())
    doc["models"]["peer|cpu|int8"] = {"bytes": 7, "ts": 0, "n": 1,
                                      "src": "measured"}
    with open(ledger, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    os.utime(ledger, (model_sizes._cache_mtime, model_sizes._cache_mtime))
    assert model_sizes._read() is model_sizes._cache      # cache looks fresh

    model_sizes.record("b", "cuda", "float16", 2 * GB)
    models = json.loads(open(ledger, encoding="utf-8").read())["models"]
    assert set(models) >= {"a|cuda|float16", "peer|cpu|int8", "b|cuda|float16"}


def test_record_holds_the_cross_process_save_lock(ledger, monkeypatch):
    entered = record_calls(monkeypatch, atomic_json, "save_lock", lambda path: path)
    model_sizes.record("a", "cuda", "float16", 1 * GB)
    # Once around the whole read-modify-write — never re-acquired inside
    # the write (the per-path lock is not reentrant).
    assert entered == [ledger]


def test_lookup_reports_source(ledger, monkeypatch):
    """lookup() says WHERE a size came from: a measurement of this placement
    (measured) or of another one (proxy), else a disk prior — the live walk,
    or a stored row when the walk finds nothing (disk); None when nothing is
    known."""
    model_sizes.record("large-v3", "cuda", "float16", 3 * GB)
    model_sizes.record("medium", "cpu", "int8", 1 * GB, measured=False)
    got = model_sizes.lookup("large-v3", "cuda", "float16")
    assert (got["bytes"], got["src"], got["n"]) == (3 * GB, "measured", 1)
    assert model_sizes.lookup("medium", "cpu", "int8")["src"] == "disk"
    assert model_sizes.lookup("large-v3", "cpu", "int8")["src"] == "proxy"
    monkeypatch.setattr(model_sizes, "disk_size",
                        lambda name: 2 * GB if name == "on-disk" else None)
    assert model_sizes.lookup("on-disk", "cuda", "float16") == {
        "bytes": 2 * GB, "src": "disk", "n": 0, "ts": None}
    assert model_sizes.lookup("never", "cuda", "float16") is None
    assert model_sizes.estimate("on-disk", "cuda", "float16") == 2 * GB


def test_proxy_prefers_a_same_device_measurement_over_a_disk_row(ledger):
    """The ledger is written sorted, so a first-match fallback always picked
    the cpu row: a cpu DISK prior shadowed a cuda MEASUREMENT of the same
    model and preload deferred a model that fits."""
    model_sizes.record("large-v3", "cuda", "int8_float16", 2 * GB)
    model_sizes.record("large-v3", "cpu", "int8", 6 * GB, measured=False)
    assert _bytes_src("large-v3", "cuda", "float16") == (2 * GB, "proxy")
    # Same device family beats another device; a disk row comes last.
    model_sizes.record("large-v3", "cuda", "float32", 5 * GB)
    assert model_sizes.lookup("large-v3", "cuda", "float16")["bytes"] == 5 * GB
    assert model_sizes.lookup("large-v3", "cpu", "int16")["bytes"] == 5 * GB
    # A stray cpu "measured" row (an NVML delta an older build recorded)
    # ranks like a disk row, never above a cuda measurement.
    model_sizes.record("large-v3", "cpu", "float32", 9 * GB)
    assert _bytes_src("large-v3", "cpu", "int16") == (5 * GB, "proxy")


def test_a_live_disk_walk_beats_the_exact_placements_disk_row(
        ledger, tmp_path, monkeypatch):
    """The exact placement's disk prior is no better than a fresh walk
    either: an inflated row an older build wrote must not stand in for
    the real size (preload would defer a model that fits). It is kept
    only for a model no longer on disk; a measurement still wins
    outright."""
    import shutil
    blobs = _hf_repo(tmp_path, monkeypatch, "org--repo") / "blobs"
    write_bytes(blobs / "a", b"x" * 3000)
    model_sizes.record("pyannote:org/repo", "cuda", "float32", 30_000,
                       measured=False)
    assert _bytes_src("pyannote:org/repo", "cuda", "float32") == (3000, "disk")
    shutil.rmtree(tmp_path / "hf")
    assert _bytes_src("pyannote:org/repo", "cuda", "float32") == (30_000, "disk")
    model_sizes.record("pyannote:org/repo", "cuda", "float32", 50_000)
    assert _bytes_src("pyannote:org/repo", "cuda", "float32") == (50_000, "measured")


def test_a_live_disk_walk_beats_another_placements_disk_row(ledger,
                                                             monkeypatch):
    """disk_size() is name-only, so a stored disk row of ANOTHER placement
    is never better than a fresh walk, only staler: an inflated prior an
    older build wrote under a placement that is never loaded again must
    not defer a model that fits. The stored row stands in only when the
    walk finds nothing."""
    model_sizes.record("m", "cpu", "int8", 2 * GB, measured=False)
    monkeypatch.setattr(model_sizes, "disk_size", lambda name: 1 * GB)
    assert _bytes_src("m", "cuda", "float16") == (1 * GB, "disk")
    monkeypatch.setattr(model_sizes, "disk_size", lambda name: None)
    assert _bytes_src("m", "cuda", "float16") == (2 * GB, "disk")


def test_a_newer_disk_prior_replaces_an_older_one(ledger):
    """A disk walk is deterministic: an inflated prior an older build wrote
    (symlink double count, every GGUF quant) must give way to the
    corrected walk, not survive as a high-water mark."""
    model_sizes.record("x", "cpu", "int8", 2000, measured=False)
    model_sizes.record("x", "cpu", "int8", 1000, measured=False)
    assert _bytes_src("x", "cpu", "int8") == (1000, "disk")


def test_a_disk_prior_replaces_a_stray_cpu_measurement(ledger):
    """NVML cannot measure a cpu load; a cpu row an older build wrote as
    "measured" (somebody else's VRAM delta) must not outrank the disk
    prior forever, since a cpu load never produces a measurement again."""
    model_sizes.record("y", "cpu", "int8", 300)
    model_sizes.record("y", "cpu", "int8", 5000, measured=False)
    assert _bytes_src("y", "cpu", "int8") == (5000, "disk")


def test_an_exact_stray_cpu_measurement_reads_as_disk(ledger):
    """The exact-placement hit applies the same provenance rule as the
    fallback: a cpu row an older build wrote as "measured" is a VRAM delta,
    not a measurement of the cpu load, so /stats must not call it one."""
    model_sizes.record("z", "cpu", "int8", 300)
    assert _bytes_src("z", "cpu", "int8") == (300, "disk")


def test_disk_size_sizes_a_local_model_directory(ledger, tmp_path):
    """DEFAULT_MODEL may be a local CT2 directory the loader opens in place;
    it must be sized there, not looked up as a never-existing hub leaf."""
    (tmp_path / "model.bin").write_bytes(b"x" * 1234)
    assert model_sizes.disk_size(str(tmp_path)) == 1234


def test_tilde_roots_resolve_under_the_home_dir(ledger, tmp_path,
                                                 monkeypatch):
    """The hub expands "~" in HF_HOME and in an explicit cache_dir; a .env
    value is not expanded by python-dotenv, so the lookup must do it."""
    from faster_whisper_backend.runtime import hf_cache
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HF_HOME", "~/hf")
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", "~/dl", raising=False)
    assert hf_cache.hub_cache_dir().startswith(str(tmp_path))
    assert hf_cache.hub_lookup_dir().startswith(str(tmp_path))
    assert model_sizes._model_path("gguf:o/r:Q4").startswith(str(tmp_path))
    monkeypatch.delenv("HF_HOME")
    # normpath: on Windows expanduser keeps the "/" of "~/dl".
    assert os.path.normpath(hf_cache.hub_cache_dir()) == os.path.join(
        str(tmp_path), "dl", "hf", "hub")
    assert model_sizes._model_path("openai/whisper-tiny").startswith(
        str(tmp_path))


def test_app_module_keeps_record_off_the_default_ledger(app_module, tmp_path,
                                                        monkeypatch):
    # record() is reached through a model_registry registration (a positive
    # VRAM delta is a measurement); its lock, read and write must all land
    # on app_module's tmp ledger, never on the import-time default path.
    from faster_whisper_backend.runtime import model_registry
    written = record_calls(monkeypatch, atomic_json, "atomic_write_json",
                           lambda doc, path, **kw: path)
    model_registry.register_loaded_model("iso", 1024, "cuda", "int8")
    assert written == [str(tmp_path / "model_sizes.json")]
    doc = json.loads((tmp_path / "model_sizes.json").read_text(encoding="utf-8"))
    assert "iso|cuda|int8" in doc["models"]
