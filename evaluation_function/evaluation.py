"""
evaluation.py
=============
Lambda Feedback platform calls evaluation_function(response, answer, params) 
and expects a dict back with at least "is_correct" and "feedback" keys.
All evaluation logic is in compare_MIDI.py, 
all audio processing logic is in audio_processing.py, 
this file is for the platform interface.
"""


from typing import Any
from lf_toolkit.evaluation import Result, Params

from .compare_MIDI import (
    compare_performance_ED,
    DEFAULT_GAP_PENALTY,
    TIMING_RELATIVE_THRESHOLD,
    DURATION_RELATIVE_THRESHOLD,
    GLOBAL_SLOW_THRESHOLD,
    GLOBAL_FAST_THRESHOLD,
    DEFAULT_CHORD_ONSET_WINDOW
)
from .audio_processing import is_audio_input, load_basic_pitch_model, transcription_pipeline

# Load the AMT model once when this module is imported (cold start),
# not inside evaluation_function, so it is not reloaded on every request.
BASIC_PITCH_MODEL = load_basic_pitch_model()


def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Result:
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

    if is_audio_input(response):
        # Step 1a: audio -> notes, only when needed
        compare_midi_input, predicted_notes, runtime_seconds = transcription_pipeline(
            response,
            BASIC_PITCH_MODEL,
            apply_postprocessing=params.get("apply_postprocessing", True),
        )
    else:
        # response is already in the notes format compare_MIDI expects
        compare_midi_input = response
    
    result = compare_performance_ED(
        compare_midi_input,
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