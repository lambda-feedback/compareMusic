"""
pianovam_evaluation.py
===================
Helper functions used to evaluate Basic Pitch transcriptions against the PianoVAM ground truth.
"""

import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import pandas as pd

from evaluation_function.audio_processing import transcribe_audio_cached
from evaluation_function.compare_MIDI import compare_performance_ED


def get_sample_paths(sample, audio_dir, midi_dir, tsv_dir):
    """
    Build the audio/MIDI/TSV file paths for one PianoVAM sample.
    """
    file_stem = str(sample["record_time"])
    return {
            "audio": audio_dir / f"{file_stem}.wav",
            "midi": midi_dir / f"{file_stem}.mid",
            "tsv": tsv_dir / f"{file_stem}.tsv",
        }


def load_ground_truth_notes(tsv_path):
    """
    Load PianoVAM ground truth notes from a TSV annotation file.
    """
    tsv_df = pd.read_csv(tsv_path, sep="\t")
    notes = []

    for _, row in tsv_df.iterrows():
        onset = float(row["# onset"])
        key_offset = float(row["key_offset"])
        frame_offset = float(row["frame_offset"])
        notes.append({
            "pitch": int(row["note"]),
            "onset": onset,
            "key_offset": key_offset,
            "frame_offset": frame_offset,
            "key_duration": key_offset - onset,
            "audible_duration": frame_offset - onset,
            "velocity": int(row["velocity"]),
        })

    notes.sort(key=lambda note: (note["onset"], note["pitch"]))
    return notes


def notes_to_mir_eval_arrays(notes, offset_key="offset"):
    """
    Convert a list of note dictionaries into the (intervals,
    frequencies) arrays expected by mir_eval.
    """
    intervals = np.array(
        [[note["onset"], note[offset_key]] for note in notes],
        dtype=float,
    )

    midi_pitches = np.array(
        [note["pitch"] for note in notes],
        dtype=float,
    )

    frequencies = 440.0 * (2.0 ** ((midi_pitches - 69.0) / 12.0))

    return intervals, frequencies


def evaluate_note_transcription(reference_notes, predicted_notes, onset_tolerance=0.05):
    """
    Compare predicted notes against ground truth notes using
    mir_eval's standard transcription metrics (precision, recall, F1,
    both with and without requiring a matching offset).

    Ground truth duration uses frame_offset (audible duration
    including pedal decay), not key_offset (raw key release), because
    we are evaluating an audio-only AMT model against what could
    actually be heard.
    """
    reference_intervals, reference_pitches = notes_to_mir_eval_arrays(
        reference_notes,
        offset_key="frame_offset",
    )

    predicted_intervals, predicted_pitches = notes_to_mir_eval_arrays(
        predicted_notes,
        offset_key="offset",
    )

    precision, recall, f1, overlap = mir_eval.transcription.precision_recall_f1_overlap(
        reference_intervals,
        reference_pitches,
        predicted_intervals,
        predicted_pitches,
        onset_tolerance=onset_tolerance,
    )

    precision_no_offset, recall_no_offset, f1_no_offset, overlap_no_offset = mir_eval.transcription.precision_recall_f1_overlap(
        reference_intervals,
        reference_pitches,
        predicted_intervals,
        predicted_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=None,
    )

    return {
        "note_precision": precision,
        "note_recall": recall,
        "note_f1": f1,
        "note_precision_no_offset": precision_no_offset,
        "note_recall_no_offset": recall_no_offset,
        "note_f1_no_offset": f1_no_offset,
        "note_average_overlap_with_offset": overlap,
        "reference_notes": len(reference_notes),
        "predicted_notes": len(predicted_notes),
        "note_count_difference": len(predicted_notes) - len(reference_notes),
    }


def build_compare_midi_input(notes, duration_key):
    """
    Convert notes into the {"notes": [...]} format expected by
    compare_performance_ED.
    """
    converted_notes = []

    for note in notes:
        converted_notes.append({
            "pitch": note["pitch"],
            "start": note["onset"],
            "duration": note[duration_key],
        })

    return {"notes": converted_notes}


def analyse_transcription_output(reference_notes, predicted_notes, chord_onset_window=0.05):
    """
    Run the compareMusic alignment pipeline (compare_performance_ED)
    on a transcription result, and return a flat dictionary of stats.
    """
    response_midi = build_compare_midi_input(predicted_notes, "duration")
    reference_midi = build_compare_midi_input(reference_notes, "audible_duration")

    result = compare_performance_ED(
        response_midi,
        reference_midi,
        chord_onset_window=chord_onset_window,
    )

    stats = result.stats

    return {
        "transcription_missing_notes": stats["total_notes_missing"],
        "transcription_extra_notes": stats["total_notes_extra"],
        "transcription_wrong_pitch_notes": stats["total_notes_wrong_pitch"],
        "transcription_wrong_timing_notes": stats["total_notes_wrong_timing"],
        "transcription_wrong_duration_notes": stats["total_notes_wrong_duration"],
        "transcription_missing_chords": stats["total_chords_missing"],
        "transcription_extra_chords": stats["total_chords_extra"],
        "transcription_imperfect_chords": stats["total_chords_imperfect"],
        "transcription_wrong_chords": stats["total_chords_wrong"],
        "transcription_output_is_correct": result.is_correct,
    }


def evaluate_on_pianovam(
    split_name,
    metadata_df,
    audio_dir,
    midi_dir,
    tsv_dir,
    cache_dir,
    basic_pitch_model,
):
    """
    Run transcription + evaluation over every "Solo" sample in one
    PianoVAM split (e.g. "train" or "test"), and return one row of
    results per sample as a DataFrame.
    """
    split_samples = metadata_df[
        (metadata_df["split"] == split_name)
        & (metadata_df["performance_method"] == "Solo")
    ].copy()

    rows = []

    for row_index, sample in split_samples.iterrows():
        paths = get_sample_paths(sample, audio_dir, midi_dir, tsv_dir)
        reference_notes = load_ground_truth_notes(paths["tsv"])
        predicted_notes, runtime_seconds = transcribe_audio_cached(
            cache_dir, sample["sample_id"], paths["audio"], basic_pitch_model
        )

        transcription_metrics = evaluate_note_transcription(reference_notes, predicted_notes)
        transcription_output = analyse_transcription_output(reference_notes, predicted_notes)

        result = {
            "sample_id": sample["sample_id"],
            "record_time": sample["record_time"],
            "composer": sample["composer"],
            "piece": sample["piece"],
            "duration_seconds": sample["duration_seconds"],
        }
        result.update(transcription_metrics)
        result.update(transcription_output)
        result["runtime_seconds"] = runtime_seconds

        rows.append(result)

    return pd.DataFrame(rows)


def plot_transcription_piano_roll(ground_truth_notes, predicted_notes, title=None):
    """
    Plot the Basic Pitch transcription against the PianoVAM ground
    truth, shown three ways: predicted notes, ground truth key
    duration, and ground truth audible duration (with pedal).
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    if title is not None:
        fig.suptitle(title)

    all_pitches = (
        [note["pitch"] for note in predicted_notes]
        + [note["pitch"] for note in ground_truth_notes]
    )

    pitch_min = min(all_pitches) - 1
    pitch_max = max(all_pitches) + 1

    all_times = (
        [note["offset"] for note in predicted_notes]
        + [note["frame_offset"] for note in ground_truth_notes]
    )

    time_max = max(all_times)

    for note in predicted_notes:
        axes[0].plot(
            [note["onset"], note["offset"]],
            [note["pitch"], note["pitch"]],
            linewidth=3,
            alpha=0.8,
        )

    axes[0].set_title("Basic Pitch transcription")
    axes[0].set_ylabel("MIDI pitch")
    axes[0].grid(alpha=0.15)

    for note in ground_truth_notes:
        axes[1].plot(
            [note["onset"], note["key_offset"]],
            [note["pitch"], note["pitch"]],
            linewidth=3,
            alpha=0.8,
        )

    axes[1].set_title("PianoVAM ground truth: key duration")
    axes[1].set_ylabel("MIDI pitch")
    axes[1].grid(alpha=0.15)

    for note in ground_truth_notes:
        axes[2].plot(
            [note["onset"], note["frame_offset"]],
            [note["pitch"], note["pitch"]],
            linewidth=3,
            alpha=0.8,
        )

    axes[2].set_title("PianoVAM ground truth: audible duration including pedal")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("MIDI pitch")
    axes[2].grid(alpha=0.15)

    for ax in axes:
        ax.set_ylim(pitch_min, pitch_max)
        ax.set_xlim(0, time_max)

    plt.tight_layout()
    plt.show()