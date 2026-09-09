"""
transport_test.py
=================
Regression tests for the transport that carries evaluation requests from
Shimmy to this function.

The Lambda Feedback platform sends `response` and `answer` as JSON, and a
real piece of music easily runs to thousands of bytes. Requests travel over
a unix socket using lf_toolkit's newline-delimited framing, so a framing
bug that truncates long messages silently breaks every realistic
submission while short test cases keep passing.

These tests exercise lf_toolkit's framing directly with a payload larger
than the 4096-byte read hint, so that a regression in the pinned toolkit
version is caught here rather than in production.

Run locally with:  python -m pytest evaluation_function/transport_test.py -v
"""

import json

import anyio

from lf_toolkit.io.stream_io import NewlineStreamIO

from .evaluation_test import make_midi


# Helpers
# ------------------------------------------------------------------------------
class FakeStream:
    """
    Minimal in-memory stand-in for a socket.

    read(size) returns at most `size` bytes from the pending buffer, exactly
    as a real socket may return a short read.
    """

    def __init__(self, data=b""):
        self._pending = data
        self.written = b""

    async def read(self, size):
        chunk = self._pending[:size]
        self._pending = self._pending[size:]
        return chunk

    async def write(self, data):
        self.written += data


def make_long_request(note_count):
    """
    Build a realistic evaluation request of roughly `note_count` notes,
    serialised the way the platform sends it.
    """
    pitches = [60 + (i % 12) for i in range(note_count)]
    starts = [round(i * 0.5, 3) for i in range(note_count)]
    durations = [0.4] * note_count

    midi = make_midi(pitches, starts, durations)
    return json.dumps({"response": midi, "answer": midi, "params": {}})


def round_trip(payload):
    """
    Write `payload` through NewlineStreamIO and read it back, the way the
    IPC server frames one request.
    """

    async def run():
        writer_stream = FakeStream()
        await NewlineStreamIO(writer_stream).write(payload.encode("utf-8"))

        reader = NewlineStreamIO(FakeStream(writer_stream.written))
        return await reader.read(4096)

    return anyio.run(run)


# Tests
# ------------------------------------------------------------------------------
def test_short_request_round_trips():
    """A request comfortably under the read hint must survive unchanged."""
    payload = make_long_request(5)
    assert len(payload) < 4096

    assert round_trip(payload).decode("utf-8") == payload


def test_request_larger_than_read_hint_is_not_truncated():
    """
    A request larger than the 4096-byte read hint must survive unchanged.

    The framing is newline-delimited, so the read must continue to the
    delimiter rather than stopping at the size hint.
    """
    payload = make_long_request(200)
    assert len(payload) > 4096, "test payload must exceed the read hint"

    assert round_trip(payload).decode("utf-8") == payload


def test_large_request_still_parses_as_json():
    """
    The practical symptom of truncation: the worker receives a prefix of the
    request and cannot decode it, so it never replies and the caller times out.
    """
    payload = make_long_request(200)

    received = round_trip(payload).decode("utf-8")
    parsed = json.loads(received)

    assert len(parsed["response"]["notes"]) == 200
    assert len(parsed["answer"]["notes"]) == 200
