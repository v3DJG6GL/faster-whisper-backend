"""metrics.classify_error: the six-word vocabulary the failures card groups
by. Message-based where the libraries give nothing better, so the exact
strings matched are pinned here."""

import asyncio

import pytest

import metrics
import url_download


class _TorchOOM(Exception):
    pass


_TorchOOM.__name__ = "OutOfMemoryError"


class _AvError(Exception):
    pass


_AvError.__module__ = "av.error"


class _Http(Exception):
    def __init__(self, code):
        super().__init__("http")
        self.status_code = code


@pytest.mark.parametrize("exc,status,stage,expected", [
    (None, "ok", "transcribing", (None, None)),
    (None, "cancelled", "diarizing", ("cancelled", "diarizing")),
    (asyncio.CancelledError(), "error", None, ("cancelled", None)),
    (_TorchOOM("CUDA out of memory"), "error", "transcribing", ("cuda_oom", "transcribing")),
    (RuntimeError("CUDA failed with error out of memory"), "error", "transcribing",
     ("cuda_oom", "transcribing")),
    (RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"), "error", "separating", ("cuda_oom", "separating")),
    (RuntimeError("[ONNXRuntimeError] Failed to allocate memory for requested buffer"),
     "error", "diarizing", ("cuda_oom", "diarizing")),
    (MemoryError("out of memory"), "error", "transcribing", ("cuda_oom", "transcribing")),
    (asyncio.TimeoutError(), "error", "downloading", ("timeout", "downloading")),
    (TimeoutError("slow"), "error", "translating", ("timeout", "translating")),
    (RuntimeError("the download timed out"), "error", "downloading", ("timeout", "downloading")),
    (url_download.UrlPolicyError("this site isn't allowed by the server's URL policy"),
     "error", "downloading", ("policy_blocked", "downloading")),
    (url_download.UrlTimeoutError("the download timed out"), "error", "downloading",
     ("timeout", "downloading")),
    (url_download.UrlDownloadError("the URL has no host"), "error", "downloading",
     ("other", "downloading")),
    (_AvError("Invalid data found when processing input"), "error", "transcribing",
     ("decode_failed", "transcribing")),
    (ValueError("bad"), "error", "analyzing", ("decode_failed", "analyzing")),
    (_Http(499), "cancelled", None, ("cancelled", None)),
    (_Http(499), "error", "transcribing", ("cancelled", "transcribing")),
    (_Http(400), "error", "downloading", ("rejected", "downloading")),
    (_Http(413), "error", None, ("rejected", None)),
    (_Http(500), "error", "transcribing", ("other", "transcribing")),
    (RuntimeError("decode blew up"), "error", "transcribing", ("other", "transcribing")),
    (None, "error", "transcribing", ("other", "transcribing")),
])
def test_classify_error(exc, status, stage, expected):
    assert metrics.classify_error(exc, status=status, stage=stage) == expected


def test_error_classes_are_the_documented_six_plus_other():
    assert metrics.ERROR_CLASSES == ("policy_blocked", "cuda_oom", "timeout",
                                     "cancelled", "decode_failed", "rejected", "other")
    assert url_download.UrlPolicyError.error_class in metrics.ERROR_CLASSES
    assert url_download.UrlTimeoutError.error_class in metrics.ERROR_CLASSES
