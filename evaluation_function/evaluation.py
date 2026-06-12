from typing import Any
from lf_toolkit.evaluation import Result, Params
import difflib


def basic_comparison(responseMIDI, 
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

    feedback = []
    all_correct = True
    
    # match the pitches to find if the student play extra or missing notes during practice
    ref_pitches = [note["pitch"] for note in ref_notes]
    response_pitches = [note["pitch"] for note in response_notes]
    pitch_similarity = difflib.SequenceMatcher(None, ref_pitches, response_pitches)

    for op, ref_start, ref_end, response_start, response_end in pitch_similarity.get_opcodes():

        # if the pitches are the same, then check the timing and duration
        if op == 'equal': 
            for i in range(ref_end - ref_start):
                ref_note = ref_notes[ref_start + i]
                response_note = response_notes[response_start + i]

                timing_difference = abs(ref_note["start"] - response_note["start"])
                duration_difference = abs(ref_note["duration"] - response_note["duration"])
                timing_match = timing_difference <= timing_tolerance
                duration_match = duration_difference <= duration_tolerance

                if timing_match and duration_match:
                    feedback.append(
                        f"Note {ref_start+i+1} with pitch {ref_note['pitch']} is correct.")
                else:
                    all_correct = False
                    if not timing_match:
                        feedback.append(f"Note {ref_start+i+1}: difference in start time: {timing_difference:.2f}s.")
                    if not duration_match:
                        feedback.append(f"Note {ref_start+i+1}: difference in duration: {duration_difference:.2f}s.")

        # if the pitches are different, then check which pitch is wrong and give feedback
        elif op == 'replace':
            all_correct = False
            for i in range(ref_end - ref_start):
                ref_note = ref_notes[ref_start + i]
                response_note = response_notes[response_start + i]
                feedback.append(f"Note {ref_start+i+1} is wrong: expected {ref_note['pitch']}, but played {response_note['pitch']}.")

        # if some notes are missing, then give feedback about which notes are missing     
        elif op == 'delete':
            all_correct = False
            for i in range(ref_end - ref_start):
                ref_note = ref_notes[ref_start + i]
                feedback.append(f"Note {ref_start+i+1} with pitch {ref_note['pitch']} is missing in your performance.")

        # if some extra notes are played, then give feedback about which extra notes are played
        elif op == 'insert':
            all_correct = False
            for i in range(response_end - response_start):
                response_note = response_notes[response_start + i]
                feedback.append(f"You played an extra note {response_start+i+1} with pitch {response_note['pitch']}.")

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
    all_correct, feedback = basic_comparison(response, answer)

    return Result(
    is_correct=all_correct,
    feedback_items=[("feedback", "\n".join(feedback))]
    )