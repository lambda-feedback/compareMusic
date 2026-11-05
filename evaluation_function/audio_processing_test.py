"""
audio_processing_test.py
========================
Unit tests for reading audio input. The platform sends uploaded
recordings as URLs to a bucket, so these cover URLs as well as the
local paths used in development.

Sections
--------
1. Tests for file_extension, file_name and is_url
2. Tests for is_audio_input
3. Tests for local_file
4. Tests for reading MIDI files
5. Tests that every supported audio format decodes
"""


import contextlib
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .audio_processing import (
    file_extension,
    file_name,
    is_url,
    AUDIO_EXTENSIONS,
    is_audio_input,
    is_midi_file_input,
    local_file,
    midi_file_to_notes,
)


# A presigned bucket URL: the extension is followed by a query string.
SIGNED_URL = "https://bucket.example.com/uploads/rec.wav?X-Amz-Signature=abc&expires=1"


# 1. Tests for file_extension, file_name and is_url
# ------------------------------------------------------------------------------
class TestPathHelpers(unittest.TestCase):

    def test_extension_of_local_path(self):
        assert file_extension("/uploads/rec.wav") == ".wav"

    def test_extension_ignores_query_string(self):
        assert file_extension(SIGNED_URL) == ".wav"

    def test_extension_ignores_fragment(self):
        assert file_extension("https://x.example.com/rec.mp3#part2") == ".mp3"

    def test_extension_is_lowercased(self):
        assert file_extension("/uploads/REC.WAV") == ".wav"

    def test_name_drops_the_signature(self):
        assert file_name(SIGNED_URL) == "rec.wav"

    def test_is_url_only_for_http(self):
        assert is_url("https://x.example.com/rec.wav") is True
        assert is_url("http://x.example.com/rec.wav") is True
        assert is_url("/uploads/rec.wav") is False
        assert is_url({"notes": []}) is False


# 2. Tests for is_audio_input
# ------------------------------------------------------------------------------
class TestIsAudioInput(unittest.TestCase):

    def test_signed_url_is_recognised_as_audio(self):
        assert is_audio_input(SIGNED_URL) is True

    def test_local_audio_path_is_recognised(self):
        assert is_audio_input("/uploads/rec.flac") is True

    def test_notes_dict_is_not_audio(self):
        assert is_audio_input({"notes": []}) is False

    def test_midi_url_is_not_audio(self):
        assert is_audio_input("https://x.example.com/rec.mid?sig=abc") is False

    def test_every_supported_audio_extension_is_recognised(self):
        for extension in [".mp3", ".wav", ".ogg", ".aac", ".flac", ".opus", ".m4a"]:
            with self.subTest(extension=extension):
                assert is_audio_input(f"https://x.example.com/rec{extension}?sig=abc") is True


# 3. Tests for local_file
# ------------------------------------------------------------------------------
class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@contextlib.contextmanager
def serve_directory(directory):
    """Serve directory over HTTP on a free local port; yield its base URL."""
    handler = partial(QuietHandler, directory=directory)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


class TestLocalFile(unittest.TestCase):

    CONTENT = b"stand-in for audio bytes"

    def test_local_path_is_handed_back_unchanged(self):
        with local_file("/uploads/rec.wav") as path:
            assert path == "/uploads/rec.wav"

    def test_url_is_downloaded_and_removed_afterwards(self):
        with tempfile.TemporaryDirectory() as served_directory:
            with open(os.path.join(served_directory, "rec.wav"), "wb") as f:
                f.write(self.CONTENT)

            handler = partial(QuietHandler, directory=served_directory)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()

            try:
                url = f"http://127.0.0.1:{server.server_port}/rec.wav?X-Amz-Signature=abc"
                with local_file(url) as path:
                    # Basic Pitch picks its decoder by extension, so the
                    # temporary file has to keep the .wav suffix.
                    assert path.endswith(".wav")
                    with open(path, "rb") as f:
                        assert f.read() == self.CONTENT
                    downloaded_path = path

                assert not os.path.exists(downloaded_path)
            finally:
                server.shutdown()


# 4. Tests for reading MIDI files
# ------------------------------------------------------------------------------
def write_midi(path, notes):
    """Write (pitch, start, end) tuples to a MIDI file, plus one drum hit."""
    import pretty_midi

    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    for pitch, start, end in notes:
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end))
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    drums.notes.append(pretty_midi.Note(velocity=80, pitch=36, start=0.0, end=0.1))
    midi.instruments.extend([piano, drums])
    midi.write(path)


class TestMidiFiles(unittest.TestCase):

    NOTES = [(64, 0.5, 1.0), (60, 0.0, 0.5)]

    def test_midi_paths_and_urls_are_recognised(self):
        assert is_midi_file_input("/uploads/rec.mid") is True
        assert is_midi_file_input("https://x.example.com/rec.MIDI?sig=abc") is True
        assert is_midi_file_input("https://x.example.com/rec.wav") is False
        assert is_midi_file_input('{"notes": []}') is False
        assert is_midi_file_input({"notes": []}) is False

    def test_reads_notes_from_a_url_ignoring_drums(self):
        with tempfile.TemporaryDirectory() as directory:
            write_midi(os.path.join(directory, "rec.mid"), self.NOTES)
            with serve_directory(directory) as base_url:
                result = midi_file_to_notes(f"{base_url}/rec.mid?X-Amz-Signature=abc")

        assert [note["pitch"] for note in result["notes"]] == [60, 64]
        assert result["notes"][1]["start"] == 0.5
        assert abs(result["notes"][1]["duration"] - 0.5) < 1e-6


# 5. Tests that every supported audio format decodes
# ------------------------------------------------------------------------------
@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is needed to create and decode the samples")
class TestAudioFormatsDecode(unittest.TestCase):
    """
    Basic Pitch loads audio with librosa. Encode a short tone in each
    supported format and check librosa can read it back, so a missing
    decoder (e.g. no ffmpeg for m4a/aac) shows up here rather than on a
    student's submission.
    """

    def test_each_format_decodes(self):
        import librosa

        with tempfile.TemporaryDirectory() as directory:
            for extension in AUDIO_EXTENSIONS:
                with self.subTest(extension=extension):
                    path = os.path.join(directory, "tone" + extension)
                    subprocess.run(
                        ["ffmpeg", "-loglevel", "error", "-f", "lavfi",
                         "-i", "sine=frequency=440:duration=0.5", path],
                        check=True,
                    )
                    samples, _ = librosa.load(path, sr=22050, mono=True)
                    assert len(samples) > 0
