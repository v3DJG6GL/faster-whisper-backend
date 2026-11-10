"""system_stats.available_ram_bytes: psutil's available RAM, capped by what
the process's memory cgroup still allows. psutil reads /proc/meminfo, the
HOST's inside a container, so the RAM ceilings (upload decode, separator,
model fit) would otherwise size from memory the container may not use.

The cgroup files are faked through system_stats._read_text: the box running
the suite may sit under any limit, or none."""

import errno

import psutil
import pytest

from faster_whisper_backend.runtime import model_sizes
from faster_whisper_backend.runtime import system_stats

GB = 1_000_000_000


def _vm(monkeypatch, available, total=64 * GB):
    real = psutil.virtual_memory
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: real()._replace(available=available,
                                                total=total))


def _files(monkeypatch, files, *, platform="linux"):
    """Serve `files` ({path: text}) to the helper; any other path is
    missing. Returns the paths read."""
    read = []

    def _read(path):
        read.append(path)
        if path not in files:
            raise FileNotFoundError(errno.ENOENT, "no such file", path)
        if isinstance(files[path], Exception):
            raise files[path]
        return files[path]
    monkeypatch.setattr(system_stats, "_read_text", _read)
    monkeypatch.setattr(system_stats.sys, "platform", platform)
    return read


def _v2(rel="/", **dirs):
    """cgroup v2 files: dirs maps a dir under /sys/fs/cgroup ("" = the
    root) to (memory.max, memory.current, inactive_file)."""
    files = {"/proc/self/cgroup": f"0::{rel}\n"}
    for d, (limit, current, inactive) in dirs.items():
        base = "/sys/fs/cgroup" + (f"/{d}" if d else "")
        files[base + "/memory.max"] = f"{limit}\n"
        files[base + "/memory.current"] = f"{current}\n"
        files[base + "/memory.stat"] = (
            f"anon 1\nfile 2\ninactive_file {inactive}\nactive_file 3\n")
    return files


def test_a_container_limit_caps_the_host_figure(monkeypatch):
    _vm(monkeypatch, 60 * GB)
    _files(monkeypatch, _v2(**{"": (4 * GB, 3 * GB, 1 * GB)}))
    # 4 GB limit, 3 GB charged of which 1 GB is reclaimable page cache.
    assert system_stats.available_ram_bytes() == 2 * GB


def test_no_limit_keeps_psutils_number(monkeypatch):
    _vm(monkeypatch, 60 * GB)
    _files(monkeypatch, _v2(**{"": ("max", 3 * GB, 1 * GB)}))
    assert system_stats.available_ram_bytes() == 60 * GB


def test_the_tighter_figure_wins(monkeypatch):
    _vm(monkeypatch, 1 * GB)
    _files(monkeypatch, _v2(**{"": (8 * GB, 1 * GB, 0)}))
    assert system_stats.available_ram_bytes() == 1 * GB


def test_a_limit_on_an_ancestor_group_counts(monkeypatch):
    """A systemd service under a slice with MemoryMax=: the process's own
    group is unlimited, its parent is not."""
    _vm(monkeypatch, 60 * GB)
    _files(monkeypatch, _v2(
        "/sys.slice/app.service",
        **{"sys.slice/app.service": ("max", 1 * GB, 0),
           "sys.slice": (3 * GB, 2 * GB, GB // 2)}))
    assert system_stats.available_ram_bytes() == 3 * GB - (2 * GB - GB // 2)


def test_a_group_over_its_limit_has_no_headroom(monkeypatch):
    _vm(monkeypatch, 60 * GB)
    _files(monkeypatch, _v2(**{"": (4 * GB, 5 * GB, 0)}))
    assert system_stats.available_ram_bytes() == 0


def test_cgroup_v1_limit_below_the_host_counts(monkeypatch):
    _vm(monkeypatch, 60 * GB, total=64 * GB)
    _files(monkeypatch, {
        "/proc/self/cgroup": "5:cpu,cpuacct:/\n4:memory:/\n0::/\n",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": f"{4 * GB}\n",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": f"{3 * GB}\n",
        "/sys/fs/cgroup/memory/memory.stat":
            f"cache 5\ntotal_inactive_file {GB}\n",
    })
    assert system_stats.available_ram_bytes() == 2 * GB


def test_cgroup_v1_unlimited_keeps_psutils_number(monkeypatch):
    _vm(monkeypatch, 60 * GB, total=64 * GB)
    _files(monkeypatch, {
        "/proc/self/cgroup": "4:memory:/\n",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": f"{3 * GB}\n",
        "/sys/fs/cgroup/memory/memory.stat": "total_inactive_file 0\n",
    })
    assert system_stats.available_ram_bytes() == 60 * GB


@pytest.mark.parametrize("broken", [
    {"/proc/self/cgroup": PermissionError(errno.EACCES, "denied")},
    {"/sys/fs/cgroup/memory.current": PermissionError(errno.EACCES, "denied")},
    {"/sys/fs/cgroup/memory.max": "lots\n"},
    {"/sys/fs/cgroup/memory.stat": "anon 1\n"},
    {"/proc/self/cgroup": "garbage\n"},
], ids=["cgroup-list", "current", "limit-text", "no-inactive", "malformed"])
def test_unreadable_cgroup_files_fall_back_to_psutil(monkeypatch, broken):
    _vm(monkeypatch, 60 * GB)
    files = _v2(**{"": (4 * GB, 3 * GB, 1 * GB)})
    files.update(broken)
    _files(monkeypatch, files)
    assert system_stats.available_ram_bytes() == 60 * GB


def test_outside_linux_only_psutil_is_read(monkeypatch):
    _vm(monkeypatch, 60 * GB)
    read = _files(monkeypatch, _v2(**{"": (4 * GB, 3 * GB, 1 * GB)}),
                  platform="win32")
    assert system_stats.available_ram_bytes() == 60 * GB
    assert read == []


def test_the_ram_ceilings_read_the_container_limit(app_module, monkeypatch):
    """The upload decode ceiling, the separator cap and the CPU model fit
    check all size from the capped figure."""
    _vm(monkeypatch, 60 * GB)
    _files(monkeypatch, _v2(**{"": (4 * GB, 3 * GB, 1 * GB)}))
    assert app_module._upload_decode_cap_s() == 2 * GB / 2 / (16000 * 12)
    assert app_module._sep_ram_cap() == 2 * GB // 4
    monkeypatch.setattr(model_sizes, "estimate", lambda *a: 1 * GB)
    assert model_sizes.fits("m", "cpu", "int8", reserve_bytes=GB) == (
        True, None)
    assert model_sizes.fits("m", "cpu", "int8", reserve_bytes=GB + 1) == (
        False, "insufficient_ram")
