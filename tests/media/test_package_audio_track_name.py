"""The audio track's name in a packaged export.

movenc writes no `title` for an MP4 audio stream, and players read the hdlr
name as the track name instead — so an MP4 export must carry the label as
`handler_name` too, exactly as the subtitle tracks do.
"""

from __future__ import annotations

from faster_whisper_backend.media import subtitle_mux as pk


def _audio_meta(argv):
    return [argv[i + 1] for i, x in enumerate(argv) if x == "-metadata:s:a"]


def test_mp4_export_names_the_audio_track_through_handler_name():
    argv = pk.build_package_argv("/m/src.mp4", [], [], container="mp4",
                                 out_path="/w/out.mp4", default_track=None,
                                 audio_lang="de", audio_label="Deutsch (Original)")
    assert _audio_meta(argv) == ["language=deu", "title=Deutsch (Original)",
                                 "handler_name=Deutsch (Original)"]
    # Without a label the language name is the handler name as well.
    argv = pk.build_package_argv("/m/src.mp4", [], [], container="mp4",
                                 out_path="/w/out.mp4", default_track=None,
                                 audio_lang="pt-BR")
    assert _audio_meta(argv)[-1] == "handler_name=Portuguese (BR)"


def test_mkv_export_keeps_the_audio_title_only():
    argv = pk.build_package_argv("/m/src.mkv", [], [], container="mkv",
                                 out_path="/w/out.mkv", default_track=None,
                                 audio_lang="de", audio_label="Deutsch")
    assert _audio_meta(argv) == ["language=deu", "title=Deutsch"]
