"""
preview.py
==========
Preview shown to the student before they submit.

Its job is to confirm what the system read from their submission, so that
a wrong file or an empty recording is caught before it is marked. It runs
while the student is still working, so it must stay fast: in particular it
never transcribes audio, which takes seconds.

The platform's Preview type carries only "sympy" and "feedback", so the
summary goes in "feedback" as a short line of text.
"""

import json
import os
from typing import Any

from lf_toolkit.preview import Result, Params, Preview

from .audio_processing import AUDIO_EXTENSIONS
from .compare_MIDI import PITCH_CLASS_NAMES

# Fixed messages, named so that tests can assert which case was hit without
# depending on the wording, which is free to change.
NO_NOTES_MESSAGE = "No notes found in this submission."
UNREADABLE_MESSAGE = (
    "This submission could not be read as MIDI note data "
    "or as an audio recording."
)


def note_name(pitch):
    """Convert a MIDI pitch number to a name, e.g. 60 -> "C4"."""
    return PITCH_CLASS_NAMES[pitch % 12] + str(pitch // 12 - 1)


def summarise_notes(notes):
    """One line describing a list of notes: how many, how long, what range."""
    if not notes:
        return NO_NOTES_MESSAGE

    count = len(notes)
    noun = "note" if count == 1 else "notes"

    end = max(note["start"] + note["duration"] for note in notes)
    pitches = [note["pitch"] for note in notes]

    return (
        f"{count} {noun}, {end:.1f} s, "
        f"{note_name(min(pitches))} to {note_name(max(pitches))}."
    )


def preview_function(response: Any, params: Params) -> Result:
    """
    Summarise the student's submission without evaluating it.

    Args:
        response: the student's submission, as MIDI note data, a JSON string
            of the same, or the path to an audio recording.
        params: unused here, accepted for interface compatibility.

    Returns:
        Result carrying a one-line description of what was read.
    """
    try:
        # An audio recording is reported as such. Transcribing it here would
        # take seconds, which is far too slow while the student is working.
        if isinstance(response, str):
            extension = os.path.splitext(response)[1].lower()
            if extension in AUDIO_EXTENSIONS:
                name = os.path.basename(response)
                return Result(preview=Preview(
                    feedback=f"Audio recording {name}, transcribed on submission."
                ))

            response = json.loads(response)

        return Result(preview=Preview(feedback=summarise_notes(response["notes"])))

    except Exception:
        return Result(preview=Preview(feedback=UNREADABLE_MESSAGE))
