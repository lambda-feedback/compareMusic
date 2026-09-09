"""
preview_test.py
===============
Tests for the preview function.

The preview is what a student sees before submitting, so it should tell
them what the system read from their submission. The template version
echoed the response back under a "sympy" key, which is meaningless for
MIDI.

The preview must stay fast, because it runs while the student is still
working. In particular it must not transcribe audio, which takes seconds.

Run locally with:  python -m pytest evaluation_function/preview_test.py -v
"""

import json
import unittest

from .evaluation_test import make_midi
from .preview import Params, preview_function


def feedback_for(response):
    return preview_function(response, Params())["preview"]["feedback"]


class TestMidiSubmissions(unittest.TestCase):

    def test_reports_the_note_count(self):
        midi = make_midi([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4])
        assert "3 notes" in feedback_for(midi)

    def test_reports_the_duration(self):
        midi = make_midi([60, 62], [0.0, 2.0], [0.5, 0.5])
        assert "2.5" in feedback_for(midi)

    def test_reports_the_pitch_range(self):
        midi = make_midi([60, 72], [0.0, 0.5], [0.4, 0.4])
        feedback = feedback_for(midi)
        assert "C4" in feedback
        assert "C5" in feedback

    def test_accepts_a_json_string(self):
        # The platform sends response and answer as JSON strings.
        midi = make_midi([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4])
        assert "3 notes" in feedback_for(json.dumps(midi))

    def test_single_note_is_not_pluralised(self):
        midi = make_midi([60], [0.0], [0.4])
        assert "1 note" in feedback_for(midi)
        assert "1 notes" not in feedback_for(midi)


class TestSubmissionsWithoutNotes(unittest.TestCase):

    def test_empty_note_list_is_reported(self):
        assert "no notes" in feedback_for({"notes": []}).lower()

    def test_audio_path_is_described_without_transcribing(self):
        # Transcription takes seconds, which is far too slow for a preview.
        feedback = feedback_for("/uploads/practice.wav")
        assert "audio" in feedback.lower()
        assert "practice.wav" in feedback

    def test_unreadable_submission_does_not_raise(self):
        feedback = feedback_for("this is not MIDI at all")
        assert feedback

    def test_result_always_has_a_preview(self):
        for response in ({"notes": []}, "nonsense", 42, None):
            result = preview_function(response, Params())
            assert result["preview"] is not None
