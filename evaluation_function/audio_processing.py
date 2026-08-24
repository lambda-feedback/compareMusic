"""
audio_processing.py
===================
Full audio-to-MIDI transcription (AMT) pipeline built on top of
Basic Pitch, from raw audio all the way to notes ready for
compare_performance_ED.

Pipeline overview:
    1. run Basic Pitch
    2. optionally apply targeted post-processing steps
    3. return notes ready to hand to compare_MIDI.py
"""


import time
import contextlib
import io
import os

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, predict


# Parameters
# ------------------------------------------------------------------------------
# configuration values for the Basic Pitch model (decoder)
ONSET_THRESHOLD = 0.6 # Minimum amplitude of an onset activation to be considered an onset
FRAME_THRESHOLD = 0.3 # Minimum amplitude of a frame activation for a note to remain 'on'
MINIMUM_NOTE_LENGTH = 50.0 # The minimum allowed note length in frames
MELODIA_TRICK = False # Whether to use the "Melodia trick" to improve pitch estimation for monophonic instruments

# configuration values for the post-processing layer
MIN_GAP_SECONDS = 0.5
MAX_LEADING_NOTES = 3
MIN_FOLLOWING_NOTES = 5
MAX_SEARCH_SECONDS = 5.0
MAX_GAP_SECONDS = 0.02
MAX_FRAGMENT_DURATION = 0.15


# File extensions we treat as "audio that needs transcription"
AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
# File extensions we treat as "already MIDI, no transcription needed"
MIDI_EXTENSIONS = [".mid", ".midi"]


# Helper function to check if the response is audio
# ---------------------------------------------------------------------
def is_audio_input(response):
    """
    Return True if response looks like an audio file that needs to go
    through the AMT pipeline before it can be compared.

    response can be:
      - a file path string (checked by extension)
      - a dict that already has a "notes" key (already MIDI, skip AMT)
    """
    # Case 1: response is already a notes dictionary, no AMT needed
    if isinstance(response, dict) and "notes" in response:
        return False

    # Case 2: response is a file path string, check its extension
    if isinstance(response, str):
        file_extension = os.path.splitext(response)[1].lower()
        if file_extension in AUDIO_EXTENSIONS:
            return True
        if file_extension in MIDI_EXTENSIONS:
            return False

    # Default: if we cannot tell, assume it is not audio
    # (safer to fail toward the existing, well-tested MIDI path)
    return False


# Load the model
# ---------------------------------------------------------------------
def load_basic_pitch_model(model_path=ICASSP_2022_MODEL_PATH):
    """
    Load the pretrained Basic Pitch model once.
    """
    return Model(model_path)


# Running the model
# ---------------------------------------------------------------------
def convert_predicted_midi_to_notes(predicted_midi):
    """
    Convert a pretty_midi object (as returned by Basic Pitch) into a
    plain list of note dictionaries, sorted by onset then pitch.
    """
    notes = []

    for instrument in predicted_midi.instruments:
        if not instrument.is_drum:
            for midi_note in instrument.notes:
                notes.append({
                    "pitch": int(midi_note.pitch),
                    "onset": float(midi_note.start),
                    "offset": float(midi_note.end),
                    "duration": float(midi_note.end - midi_note.start),
                    "velocity": int(midi_note.velocity),
                })

    notes.sort(key=lambda note: (note["onset"], note["pitch"]))
    return notes


def transcribe_audio(
    audio_path,
    model,
    onset_threshold=ONSET_THRESHOLD,
    frame_threshold=FRAME_THRESHOLD,
    minimum_note_length=MINIMUM_NOTE_LENGTH,
    melodia_trick=MELODIA_TRICK,
):
    """
    Run Basic Pitch on one audio file and return the predicted notes.

    The threshold parameters default to the global configuration values
    above, but can be overridden per call. This makes it easy to try
    different threshold values later (e.g. in the post-processing
    experiments section) without editing this function.

    Returns (predicted_notes, predicted_midi, runtime_seconds).
    """
    start_time = time.perf_counter()
    hidden_output = io.StringIO()

    # Basic Pitch prints progress information to stdout/stderr; hide
    # it so notebook output stays readable during batch runs.
    with contextlib.redirect_stdout(hidden_output), contextlib.redirect_stderr(hidden_output):
        _, predicted_midi, _ = predict(
            str(audio_path),
            model_or_model_path=model,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            melodia_trick=melodia_trick,
            multiple_pitch_bends=False,
        )

    runtime_seconds = time.perf_counter() - start_time
    predicted_notes = convert_predicted_midi_to_notes(predicted_midi)
    return predicted_notes, predicted_midi, runtime_seconds


# post-processing
# ---------------------------------------------------------------------
def remove_leading_extra_notes(notes, 
                               min_gap_seconds=MIN_GAP_SECONDS, 
                               max_leading_notes=MAX_LEADING_NOTES, 
                               min_following_notes=MIN_FOLLOWING_NOTES, 
                               max_search_seconds=MAX_SEARCH_SECONDS):
    """
    Remove a small group of isolated predictions before the likely
    performance start.

    A leading group is removed only when:
      - it contains at most max_leading_notes;
      - a gap of at least min_gap_seconds follows it;
      - at least min_following_notes remain after the gap;
      - the possible performance start is within max_search_seconds.
    """
    # if there are no notes, return an empty list
    if len(notes) == 0:
        return []

    # sort the notes by onset time
    sorted_notes = sorted(notes, key=lambda note: note["onset"])
    start_index = 0

    # find the first index where the gap to the next note is large enough
    for i in range(len(sorted_notes) - 1):
        current_onset = sorted_notes[i]["onset"]
        next_onset = sorted_notes[i + 1]["onset"]
        gap = next_onset - current_onset

        leading_note_count = i + 1 # number of notes before the gap
        following_note_count = len(sorted_notes) - leading_note_count # number of notes after the gap

        # check if the gap is large enough
        is_large_gap = gap >= min_gap_seconds
        # ensure number of leading notes does not exceed the maximum allowed
        has_few_leading_notes = leading_note_count <= max_leading_notes
        # ensure number of following notes meets the minimum required
        has_enough_following_notes = following_note_count >= min_following_notes
        # search the first few seconds only, to avoid removing valid early notes in long performances
        is_near_beginning = next_onset <= max_search_seconds

        # if all conditions are met, move on to the next note
        if (
            is_large_gap
            and has_few_leading_notes
            and has_enough_following_notes
            and is_near_beginning
        ):
            start_index = i + 1

    processed_notes = [] # create a new list to hold the processed notes i.e. notes to be kept
    # iterate over the notes starting from the determined start index
    for note in sorted_notes[start_index:]: 
        processed_notes.append(note.copy())

    return processed_notes


def merge_same_pitch_notes(notes, 
                           max_gap_seconds=MAX_GAP_SECONDS, 
                           max_fragment_duration=MAX_FRAGMENT_DURATION):
    """
    Merge nearby same-pitch notes that appear to be fragments of one sustained note.

    This covers both:
        - a short silent gap between two fragments
        - a slight overlap caused by offset estimation

    Notes are merged only when:
        1. The gap is between -max_gap_seconds and max_gap_seconds.
        2. At least one of the two notes is no longer than
           max_fragment_duration, this helps avoid merging genuine repeated notes.
    """
    if len(notes) == 0:
        return []

    notes_by_pitch = {}
    for note in notes:
        pitch = note["pitch"]
        # create a new list for this pitch if it doesn't exist in the dictionary yet
        if pitch not in notes_by_pitch: 
            notes_by_pitch[pitch] = []
        # append a copy of the note to avoid modifying the original
        notes_by_pitch[pitch].append(note.copy())

    merged_notes = []
    for pitch_notes in notes_by_pitch.values():
        # sort the notes by onset time to ensure they are processed in order
        pitch_notes = sorted(pitch_notes, key=lambda note: note["onset"])
        current_note = pitch_notes[0].copy()

        for next_note in pitch_notes[1:]:
            gap = next_note["onset"] - current_note["offset"]
            current_duration = current_note["offset"] - current_note["onset"]
            next_duration = next_note["offset"] - next_note["onset"]
            # Check if the gap is within the allowed range
            is_close_in_time = (-max_gap_seconds <= gap <= max_gap_seconds)
            # Check if both notes are short enough to be considered a fragment
            looks_like_fragment = (
                current_duration <= max_fragment_duration
                and next_duration <= max_fragment_duration
            )
            should_merge = is_close_in_time and looks_like_fragment

            if should_merge:
                # Merge the two notes by updating the offset and duration of the current note
                new_offset = max(current_note["offset"], next_note["offset"])
                current_note["offset"] = new_offset
                current_note["duration"] = new_offset - current_note["onset"]
                # If both notes have a velocity, take the maximum to represent the merged note
                if ("velocity" in current_note) and ("velocity" in next_note):
                    current_note["velocity"] = max(current_note["velocity"], next_note["velocity"])
            else:
                merged_notes.append(current_note)
                current_note = next_note.copy()

        merged_notes.append(current_note)

    return sorted(
        merged_notes,
        key=lambda note: (note["onset"], note["pitch"])
    )


def postprocess_predictions(notes):
    """
    Combine and apply the two post-processing steps to the predicted notes.
    """
    processed_notes = remove_leading_extra_notes(notes)
    processed_notes = merge_same_pitch_notes(processed_notes)
    return processed_notes


# Connecting to compare_MIDI
# ---------------------------------------------------------------------
def build_compare_midi_input(notes, duration_key="duration"):
    """
    Convert a list of note dictionaries into the {"notes": [...]}
    format expected by compare_performance_ED.

    duration_key lets the same function handle notes that store their
    duration under different names (predicted notes use "duration";
    ground truth notes from PianoVAM use "audible_duration").
    """
    converted_notes = []

    for note in notes:
        converted_notes.append({
            "pitch": note["pitch"],
            "start": note["onset"],
            "duration": note[duration_key],
        })

    return {"notes": converted_notes}


# Full pipeline
# ---------------------------------------------------------------------
def transcription_pipeline(audio_path, model, apply_postprocessing=True):
    """
    Full production pipeline: run Basic Pitch, optionally apply post-processing, 
    and return notes ready to hand to compare_MIDI.py.

    Returns (compare_midi_input, predicted_notes, runtime_seconds).
    """
    predicted_notes, predicted_midi, runtime_seconds = transcribe_audio(audio_path, model)

    if apply_postprocessing:
        predicted_notes = postprocess_predictions(predicted_notes)

    compare_midi_input = build_compare_midi_input(predicted_notes, "duration")

    return compare_midi_input, predicted_notes, runtime_seconds