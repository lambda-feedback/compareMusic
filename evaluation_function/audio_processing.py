"""
audio_processing.py
===================
Core audio-to-MIDI transcription (AMT) pipeline built on top of Basic Pitch.

Pipeline overview:
    1. load the Basic Pitch model
    2. run the model on one audio file (or re-decode already-computed
     raw model output) into a list of note dictionaries
    3. cache raw model output on disk, so repeated experiments do not
     have to re-run the neural network
"""


from pathlib import Path
import json
import time
import contextlib
import io

import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import pandas as pd

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
from basic_pitch.inference import Model, predict, run_inference
from basic_pitch.note_creation import model_output_to_notes
from evaluation_function.compare_MIDI import compare_performance_ED
from tqdm import tqdm


# Default Basic Pitch decoding parameters.
# ------------------------------------------------------------------------------
# These can always be overridden per call, for example during a
# hyperparameter search.
ONSET_THRESHOLD = 0.5 # Minimum amplitude of an onset activation to be considered an onset
FRAME_THRESHOLD = 0.3 # Minimum amplitude of a frame activation for a note to remain 'on'
MINIMUM_NOTE_LENGTH = 127.7 # The minimum allowed note length in frames
MELODIA_TRICK = True # Whether to use the "Melodia trick" to improve pitch estimation for monophonic instruments


# Load the model
# ------------------------------------------------------------------------------
def load_basic_pitch_model(model_path=ICASSP_2022_MODEL_PATH):
    """
    Load the Basic Pitch model once.
    """
    return Model(model_path)


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


def decode_model_output(model_output, config):
    """
    Re-decode already-computed raw Basic Pitch model output using a
    new set of decoding parameters (onset_threshold, frame_threshold,
    minimum_note_length, melodia_trick).

    This is what makes the hyperparameter search fast: model_output is
    the expensive neural network result, cached once per audio file.
    Only this cheap decoding step needs to re-run for every
    configuration tried in the search.

    config must be a dictionary with keys "onset_threshold",
    "frame_threshold", "minimum_note_length", "melodia_trick".
    """
    min_note_len = int(round(
        config["minimum_note_length"] / 1000 * AUDIO_SAMPLE_RATE / FFT_HOP
    ))

    # model_output_to_notes does not modify the raw "note"/"onset"/
    # "contour" arrays in place as long as min_freq/max_freq are left
    # as None (the default) and include_pitch_bends=False. If
    # min_freq/max_freq are ever passed in, re-add a copy of
    # model_output before calling model_output_to_notes.
    _, note_events = model_output_to_notes(
        model_output,
        onset_thresh=config["onset_threshold"],
        frame_thresh=config["frame_threshold"],
        min_note_len=min_note_len,
        include_pitch_bends=False,
        melodia_trick=config["melodia_trick"],
    )

    notes = []
    for onset, offset, pitch, amplitude, _ in note_events:
        notes.append({
            "pitch": int(pitch),
            "onset": float(onset),
            "offset": float(offset),
            "duration": float(offset - onset),
            "velocity": int(round(amplitude * 127)),
        })

    notes.sort(key=lambda note: (note["onset"], note["pitch"]))
    return notes


# Caching raw Basic Pitch predictions
# ------------------------------------------------------------------------------
# Raw Basic Pitch predictions are expensive to compute (a full neural
# network forward pass) but cheap to store. Caching them to disk lets
# later experiments (hyperparameter search, post-processing
# comparisons) reuse the same predictions without re-running the model.

def raw_prediction_cache_path(cache_dir, sample_id):
    """Return the path used to cache predicted notes for one sample."""
    return Path(cache_dir) / f"{str(sample_id)}.json"


def save_predicted_notes(cache_dir, sample_id, predicted_notes):
    cache_path = raw_prediction_cache_path(cache_dir, sample_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("w") as file:
        json.dump(predicted_notes, file)


def load_predicted_notes(cache_dir, sample_id):
    cache_path = raw_prediction_cache_path(cache_dir, sample_id)
    with cache_path.open("r") as file:
        return json.load(file)


def transcribe_audio_cached(cache_dir, sample_id, audio_path, model):
    """
    Return cached predicted notes for a sample if available, otherwise
    run Basic Pitch once and cache the result.

    Returns (predicted_notes, runtime_seconds). runtime_seconds is
    None when the result came from the cache, since no transcription
    happened.

    To force a fresh transcription (for example after changing the
    decoding parameters), delete the corresponding cache file first.
    """
    cache_path = raw_prediction_cache_path(cache_dir, sample_id)

    if cache_path.exists():
        predicted_notes = load_predicted_notes(cache_dir, sample_id)
        runtime_seconds = None
        return predicted_notes, runtime_seconds

    predicted_notes, predicted_midi, runtime_seconds = transcribe_audio(audio_path, model)
    save_predicted_notes(cache_dir, sample_id, predicted_notes)
    return predicted_notes, runtime_seconds