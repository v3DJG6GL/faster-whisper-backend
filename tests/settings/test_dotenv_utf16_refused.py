"""A .env with a UTF-16 BOM stops the import instead of being decoded.

8-bit lines that a tool appended to a UTF-16 .env (Windows PowerShell 5.1 `>`
/ `Out-File`) decoded, when their byte count was even, to CJK junk: the
server started and those pins were silently gone. UTF-8, with or without a
BOM, still loads.
"""

import pytest

from tests.settings.test_secret_file_encoding import _DOTENV, _import_config_from, _printed


@pytest.mark.parametrize("data", [
    # An even-length 8-bit append: it decoded to junk before.
    b"\xff\xfe" + _DOTENV.encode("utf-16-le") + b"WHISPER_LANGUAGE=de\n",
    b"\xfe\xff" + _DOTENV.encode("utf-16-be"),
], ids=["le-bom-even-8bit-append", "be-bom"])
def test_a_utf16_dotenv_stops_import_naming_the_file_and_utf8(tmp_path, data):
    out = _import_config_from(tmp_path, data)
    assert out.returncode != 0
    assert (f"{tmp_path / '.env'} could not be read as text (it is saved as "
            "UTF-16); re-save it as UTF-8") in out.stderr, out.stderr


@pytest.mark.parametrize("data", [
    _DOTENV.encode(),
    b"\xef\xbb\xbf" + _DOTENV.encode(),
], ids=["utf8", "utf8-bom"])
def test_a_utf8_dotenv_still_loads(tmp_path, data):
    out = _import_config_from(tmp_path, data)
    assert _printed(out) == [3, "Grüße"]
