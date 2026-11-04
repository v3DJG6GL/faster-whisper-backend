"""The audio track's name in a packaged export.

movenc writes no `title` for an MP4 audio stream, and players read the hdlr
name as the track name instead — so an MP4 export must carry the label as
`handler_name` too, exactly as the subtitle tracks do.
"""

from __future__ import annotations

from tests.media.test_subtitle_mux import argv_for, audio_meta


def test_mp4_export_names_the_audio_track_through_handler_name():
    argv = argv_for(container="mp4", audio_lang="de", audio_label="Deutsch (Original)")
    assert audio_meta(argv) == ["language=deu", "title=Deutsch (Original)",
                                 "handler_name=Deutsch (Original)"]
    # Without a label the language name is the handler name as well.
    argv = argv_for(container="mp4", audio_lang="pt-BR")
    assert audio_meta(argv)[-1] == "handler_name=Portuguese (BR)"


def test_mkv_export_keeps_the_audio_title_only():
    argv = argv_for(audio_lang="de", audio_label="Deutsch")
    assert audio_meta(argv) == ["language=deu", "title=Deutsch"]


def test_mp4_handler_name_names_only_the_first_audio_track():
    """A second audio track (a commentary, a dub) keeps its own hdlr name —
    the one name an MP4 player lists — instead of the transcript's label."""
    argv = argv_for(container="mp4", audio_lang="de", audio_label="Deutsch (Original)")
    h = [(argv[i - 1], a) for i, a in enumerate(argv) if a.startswith("handler_name=")]
    assert h == [("-metadata:s:a:0", "handler_name=Deutsch (Original)")]


def test_mkv_title_and_language_tag_only_the_first_audio_track():
    """MKV shows `title` as the track name: a second audio track (a French
    dub, a commentary) keeps its own name and language, not the transcript's."""
    for container in ("mkv", "mp4"):
        argv = argv_for(container=container, audio_lang="de", audio_label="Deutsch")
        assert "-metadata:s:a" not in argv
        tags = [(argv[i - 1], a) for i, a in enumerate(argv)
                if a.startswith(("language=", "title=")) and argv[i - 1].startswith("-metadata:s:a")]
        assert tags == [("-metadata:s:a:0", "language=deu"), ("-metadata:s:a:0", "title=Deutsch")]
