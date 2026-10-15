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
3. Tests for local_audio_file
"""


import os
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .audio_processing import (
    file_extension,
    file_name,
    is_url,
    is_audio_input,
    local_audio_file,
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


# 3. Tests for local_audio_file
# ------------------------------------------------------------------------------
class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class TestLocalAudioFile(unittest.TestCase):

    CONTENT = b"stand-in for audio bytes"

    def test_local_path_is_handed_back_unchanged(self):
        with local_audio_file("/uploads/rec.wav") as path:
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
                with local_audio_file(url) as path:
                    # Basic Pitch picks its decoder by extension, so the
                    # temporary file has to keep the .wav suffix.
                    assert path.endswith(".wav")
                    with open(path, "rb") as f:
                        assert f.read() == self.CONTENT
                    downloaded_path = path

                assert not os.path.exists(downloaded_path)
            finally:
                server.shutdown()
