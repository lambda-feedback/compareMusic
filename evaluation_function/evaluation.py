from typing import Any
from lf_toolkit.evaluation import Result, Params


def compute_cost(note1, note2):
    """
    Computes the cost used for Dynamic Time Warping.
    Lower cost means the two notes are more similar.
    """
    pass

def note_alignment_DTW(responseNotes, refNotes):
    """
    Use DTW to find the optimal alignment between response and reference MIDI notes.
    """
    pass


def compare_notes(responseMIDI, 
                     refMIDI, 
                     timing_tolerance = 0.1, 
                     duration_tolerance = 0.1):
    """
    Compares student's response MIDI notes with reference MIDI notes,
    based on pitch, timing, and duration with specified tolerances.
    Args:
        refMIDI: The reference MIDI note.
        responseMIDI: The student's response MIDI note to evaluate.
        timing_tolerance: consider as correct if start is within this tolerance.
        duration_tolerance: consider as correct if duration is within this tolerance.
    Returns:
        bool: True if the notes match within the specified tolerances, False otherwise.
    """
    ref_notes = refMIDI["notes"]
    response_notes = responseMIDI["notes"]

    aligned_notes = note_alignment_DTW(response_notes, ref_notes)

    feedback = []
    all_correct = True

    # loop over each note pair
    
    
    return all_correct, feedback

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
    all_correct, feedback = compare_notes(response, answer)

    return Result(
    is_correct=all_correct,
    feedback_items=[("feedback", "\n".join(feedback))]
    )