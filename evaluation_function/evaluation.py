"""
evaluation.py
=============
Lambda Feedback platform calls evaluation_function(response, answer, params) 
and expects a dict back with at least "is_correct" and "feedback" keys.
All evaluation logic is in compare_MIDI.py, this file is for the platform interface.
"""

import json
from typing import Any, Dict
from lf_toolkit.shared.params import Params

from .compare_MIDI import (
    compare_performance_ED,
    DEFAULT_GAP_PENALTY,
    TIMING_RELATIVE_THRESHOLD,
    DURATION_RELATIVE_THRESHOLD,
    GLOBAL_SLOW_THRESHOLD,
    GLOBAL_FAST_THRESHOLD,
    DEFAULT_CHORD_ONSET_WINDOW
)
from .audio_processing import (
    AUDIO_EXTENSIONS,
    file_extension,
    file_name,
    is_audio_input,
    is_midi_file_input,
    is_uploaded_file,
    load_basic_pitch_model,
    midi_file_to_notes,
    transcription_pipeline,
    uploaded_file_extension,
)

# The Basic Pitch model is loaded lazily, on the first audio request, rather
# than at import time. The RPC worker has to bind its IPC socket within a few
# seconds of starting or shimmy stops trying to connect to it; loading the
# model (and importing basic-pitch / librosa) at import pushed that past the
# window, so every request came back as "EOF" / "Service Unavailable".
_BASIC_PITCH_MODEL = None


def get_basic_pitch_model():
    """Load the Basic Pitch model on first use and cache it for later calls."""
    global _BASIC_PITCH_MODEL
    if _BASIC_PITCH_MODEL is None:
        _BASIC_PITCH_MODEL = load_basic_pitch_model()
    return _BASIC_PITCH_MODEL


def parse_json_input(data: Any) -> Any:
    """
    Convert a JSON string into a Python object.
    """
    if isinstance(data, str):
        return json.loads(data)
    return data


class UnsupportedFileError(ValueError):
    """An uploaded file is not in a format this function can read."""


def read_file(source: str, extension: str) -> Dict[str, Any]:
    """Read an audio or MIDI file (path or URL) into note data."""
    if extension in AUDIO_EXTENSIONS:
        compare_midi_input, predicted_notes, runtime_seconds = transcription_pipeline(
            source, get_basic_pitch_model(), extension=extension
        )
        return compare_midi_input
    return midi_file_to_notes(source, extension)


def prepare_input(raw_input: Any) -> Any:
    """
    Turn one raw input (either response or answer) into the
    format that compare_performance_ED expects.

    The possible cases are:
      - raw_input is an audio file path or URL
        -> run it through the AMT (audio-to-MIDI) transcription pipeline
      - raw_input is a .mid / .midi file path or URL
        -> read the notes from the file, no transcription needed
      - raw_input is a list of uploaded files from the platform's Audio
        response area, [{"url", "name", "type", "size", "comment"}, ...]
        -> read the first file as above; an empty list has no notes
      - raw_input is already note data (a dict, or a JSON string of a dict)
        -> just parse it as JSON
    """
    if is_audio_input(raw_input) or is_midi_file_input(raw_input):
        return read_file(raw_input, file_extension(raw_input))

    data = parse_json_input(raw_input)

    if isinstance(data, list):
        if not data:
            return {"notes": []}
        # The response area allows several files, but a submission is one
        # performance, so only the first file is marked.
        data = data[0]

    if is_uploaded_file(data):
        extension = uploaded_file_extension(data)
        if not extension:
            raise UnsupportedFileError(
                f"{data.get('name') or file_name(data['url'])} is not a supported "
                "audio or MIDI file."
            )
        return read_file(data["url"], extension)

    return data


def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Dict[str, Any]:
    """
    Function used to evaluate a student response.
    ---
    The handler function passes three arguments to evaluation_function():

    - `response` which are the answers provided by the student.
    - `answer` which are the correct answers to compare against.i.e. reference
    - `params` which are any extra parameters that may be useful,
        e.g., error tolerances.

    The output of this function is what is returned as the API response
    and therefore must be JSON-encodable. It must also conform to the
    response schema.

    Any standard python library may be used, as well as any package
    available on pip (provided it is added to requirements.txt).

    The way you wish to structure you code (all in this function, or
    split into many) is entirely up to you. All that matters are the
    return types and that evaluation_function() is the main function used
    to output the evaluation response.
    """
    if params is None:
        params = {}

    # The Lambda Feedback app sends response/answer as JSON strings,
    # convert them into Python object first
    try:
        response = prepare_input(response)
        answer = prepare_input(answer)
    except UnsupportedFileError as error:
        return {"is_correct": False, "feedback": str(error)}

    result = compare_performance_ED(
        response,
        answer,
        gap_penalty=params.get("gap_penalty", DEFAULT_GAP_PENALTY),
        timing_relative_threshold=params.get(
            "timing_relative_threshold", TIMING_RELATIVE_THRESHOLD
        ),
        duration_relative_threshold=params.get(
            "duration_relative_threshold", DURATION_RELATIVE_THRESHOLD
        ),
        global_slow_threshold=params.get(
            "global_slow_threshold", GLOBAL_SLOW_THRESHOLD
        ),
        global_fast_threshold=params.get(
            "global_fast_threshold", GLOBAL_FAST_THRESHOLD
        ),
        chord_onset_window=params.get(
            "chord_onset_window", DEFAULT_CHORD_ONSET_WINDOW
        )
    )

    return {
        "is_correct": result.is_correct,
        "feedback": result.feedback_message,
    }