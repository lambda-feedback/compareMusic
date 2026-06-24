"""
evaluation.py
=============
Lambda Feedback platform calls evaluation_function(response, answer, params) 
and expects a dict back with at least "is_correct" and "feedback" keys.
All evaluation logic is in compare_music.py, this file is for the platform interface.
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
)


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
    )

    return {
        "is_correct": result.is_correct,
        "feedback": result.feedback_message,
    }