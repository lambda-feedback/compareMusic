from typing import Any
from lf_toolkit.evaluation import Result, Params


def basic_comparison(refMIDI, 
                     learnerMIDI, 
                     timing_tolerance = 0.1, 
                     duration_tolerance = 0.1):
    """
    Compares learner's MIDI notes with reference MIDI notes,
    based on pitch, timing, and duration with specified tolerances.
    Args:
        refMIDI: The reference MIDI note.
        learnerMIDI: The learner's MIDI note to evaluate.
        timing_tolerance: consider as correct if start is within this tolerance.
        duration_tolerance: consider as correct if duration is within this tolerance.
    Returns:
        bool: True if the notes match within the specified tolerances, False otherwise.
    """
    ref_notes = refMIDI["notes"]
    learner_notes = learnerMIDI["notes"]

    total_notes = len(ref_notes)
    feedbacks = []
    all_correct = True

    for i in range(total_notes):
        ref_note = ref_notes[i]
        learner_note = learner_notes[i]

        # Check pitch, timing, and duration
        pitch_match = ref_note["pitch"] == learner_note["pitch"]
        timing_match = abs(ref_note["start"] - learner_note["start"]) <= timing_tolerance
        duration_match = abs(ref_note["duration"] - learner_note["duration"]) <= duration_tolerance

        problems = []
        if not pitch_match:
            problems.append(f"Pitch {learner_note['pitch']} is incorrect, should be {ref_note['pitch']}.")
        if not timing_match:
            problems.append(f"Difference in start time: {abs(ref_note['start'] - learner_note['start']):.2f}s.")
        if not duration_match:
            problems.append(f"Difference in duration: {abs(ref_note['duration'] - learner_note['duration']):.2f}s.")
        
        if len(problems) > 0:
            feedbacks.append(" ".join(problems))
            all_correct = False
        else:
            feedbacks.append("All correct! Perfect practice!")
        
    return all_correct,feedbacks


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
    - `answer` which are the correct answers to compare against.
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
    all_correct, feedbacks = basic_comparison(response, answer)

    return Result(
        is_correct=all_correct,
        feedback=feedbacks
    )