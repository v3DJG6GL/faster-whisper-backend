"""media/subtitles.render — a transcript's cues as an SRT / WebVTT body (the
response_format=srt / vtt answer of the batch routes)."""

from faster_whisper_backend.media import subtitles


def test_srt_numbers_cues_and_uses_a_comma_before_the_milliseconds():
    doc = subtitles.render([(0.0, 1.2344, "eins"), (61.5, 3600.0, "zwei")], "srt")
    assert doc == ("1\n00:00:00,000 --> 00:00:01,234\neins\n\n"
                   "2\n00:01:01,500 --> 01:00:00,000\nzwei\n")


def test_vtt_has_the_header_no_numbers_and_a_dot_before_the_milliseconds():
    doc = subtitles.render([(0.0, 1.0, "eins"), (1.0, 2.5, "zwei")], "vtt")
    assert doc == ("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\neins\n\n"
                   "00:00:01.000 --> 00:00:02.500\nzwei\n")


def test_no_cues():
    assert subtitles.render([], "srt") == ""
    assert subtitles.render([], "vtt") == "WEBVTT\n\n"


def test_an_empty_cue_is_left_out_and_the_numbers_stay_consecutive():
    # A pipeline rule can delete a whole segment's text (filler words).
    doc = subtitles.render([(0, 1, "eins"), (1, 2, "  \n "), (2, 3, "drei")], "srt")
    assert doc.split("\n\n")[1] == "2\n00:00:02,000 --> 00:00:03,000\ndrei\n"


def test_a_paragraph_break_or_an_arrow_cannot_split_the_cue():
    # A blank line ends a cue and an "-->" line reads as the next timing.
    doc = subtitles.render([(0, 1, "erste Zeile\n\n zweite Zeile ---> weiter")], "srt")
    assert doc == "1\n00:00:00,000 --> 00:00:01,000\nerste Zeile\nzweite Zeile -> weiter\n"


def test_vtt_cue_text_is_escaped():
    doc = subtitles.render([(0, 1, "a < b && c --> d")], "vtt")
    assert doc.endswith("\na &lt; b &amp;&amp; c --&gt; d\n")


def test_an_end_before_its_start_is_pulled_up_to_it():
    assert "00:00:02,000 --> 00:00:02,000" in subtitles.render([(2.0, 1.0, "x")], "srt")
