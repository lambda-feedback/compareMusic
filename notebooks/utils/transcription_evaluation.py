"""
transcription_evaluation.py
===================
Helper functions used to evaluate transcriptions against the PianoVAM ground truth
"""
import contextlib
import io
import json
from pathlib import Path
 
import mir_eval
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
from basic_pitch.inference import run_inference
from basic_pitch.note_creation import model_output_to_notes
 
from evaluation_function.audio_processing import (
    build_compare_midi_input,
    transcribe_audio,
)
from evaluation_function.compare_MIDI import compare_performance_ED
from .pianovam_loading import (
    load_ground_truth_notes, 
    get_sample_paths,
    notes_to_mir_eval_arrays,
)
 

BASIC_PITCH_DEFAULT_CONFIG = {
    "onset_threshold": 0.5,
    "frame_threshold": 0.3,
    "minimum_note_length": 127.7,
    "melodia_trick": True,
}
 
 
# Raw-prediction caching (save time during experiments; NOT used in production)
# -----------------------------------------------------------------------
def raw_prediction_cache_path(cache_dir, sample_id):
    """
    Return the path used to cache predicted notes for one sample.
    """
    return Path(cache_dir) / (f"{sample_id}.json")

def save_predicted_notes(cache_dir, sample_id, predicted_notes):
    """
    Save predicted notes for one sample to the raw-prediction cache.
    """
    cache_path = raw_prediction_cache_path(cache_dir, sample_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as file:
        json.dump(predicted_notes, file)
 
def load_predicted_notes(cache_dir, sample_id):
    """
    Load predicted notes for one sample from the raw-prediction cache.
    """
    cache_path = raw_prediction_cache_path(cache_dir, sample_id) 
    with cache_path.open("r") as file:
        return json.load(file)
 
def transcribe_audio_cached(cache_dir, sample_id, audio_path, model, **config):
    """
    Return cached predicted notes for a sample if available, otherwise
    run Basic Pitch once (optionally with non-default decoding
    parameters passed via config, e.g. onset_threshold=0.7) and cache
    the result.
 
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
 
    predicted_notes, predicted_midi, runtime_seconds = transcribe_audio(audio_path, model, **config)
    save_predicted_notes(cache_dir, sample_id, predicted_notes)
    return predicted_notes, runtime_seconds
 
 
def run_basic_pitch_inference(audio_path, model):
    """
    Run only the neural network inference step of Basic Pitch and
    return the raw model output (before onset/frame thresholding).
 
    This is the expensive part of transcription. Use this together
    with decode_model_output when you need to try many different
    decoding parameter combinations on the same audio file, for
    example during a hyperparameter search.
    """
    hidden_output = io.StringIO()
 
    with contextlib.redirect_stdout(hidden_output), contextlib.redirect_stderr(hidden_output):
        model_output = run_inference(audio_path, model)
 
    return model_output
 
 
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
 

# Evaluation metrics
# -----------------------------------------------------------------------
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


# evaluation over a PianoVAM split set
# -----------------------------------------------------------------------
def evaluate_on_pianovam(split_name, metadata_df, audio_dir, midi_dir, 
                         tsv_dir, cache_dir, basic_pitch_model):
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
            cache_dir, sample["sample_id"], paths["audio"], basic_pitch_model, 
            **BASIC_PITCH_DEFAULT_CONFIG
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

        rows.append(result)

    return pd.DataFrame(rows)


def evaluate_new_configuration_on_pianovam(config, samples_df, audio_dir, 
                                           midi_dir, tsv_dir, basic_pitch_model):
    """
    Like evaluate_on_pianovam, but runs Basic Pitch with a specific,
    non-default decoding configuration (config is a dictionary of
    keyword arguments for transcribe_audio, e.g. onset_threshold=0.7)
    and does NOT use the raw-prediction cache, since the cache only
    stores default-configuration predictions.
 
    Returns (results_df, predictions_by_sample_id).
    """
    rows = []
    predictions_by_sample_id = {}
 
    for row_index, sample in tqdm(samples_df.iterrows(), total=len(samples_df), desc="Evaluating configuration"):
        paths = get_sample_paths(sample, audio_dir, midi_dir, tsv_dir)
        reference_notes = load_ground_truth_notes(paths["tsv"])
        predicted_notes, _, _ = transcribe_audio(
            paths["audio"], basic_pitch_model, **config
        )
 
        metrics = evaluate_note_transcription(reference_notes, predicted_notes)
        transcription_output = analyse_transcription_output(reference_notes, predicted_notes)
 
        result = {"sample_id": sample["sample_id"]}
        result.update(metrics)
        result.update(transcription_output)
        rows.append(result)
 
        predictions_by_sample_id[str(sample["sample_id"])] = predicted_notes
 
    return pd.DataFrame(rows), predictions_by_sample_id


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


# summaries, comparison tables, and case studies, visualisation
# -----------------------------------------------------------------------
def summarise_amt_results(results_df):
    """
    Return mean/median/std/min/max across samples for the core
    transcription metrics.
    """
    metric_columns = [
        "note_precision",
        "note_recall",
        "note_f1",
        "note_precision_no_offset",
        "note_recall_no_offset",
        "note_f1_no_offset",
        "note_average_overlap_with_offset",
    ]
 
    summary = pd.DataFrame({
        "mean": results_df[metric_columns].mean(),
        "median": results_df[metric_columns].median(),
        "std": results_df[metric_columns].std(),
        "min": results_df[metric_columns].min(),
        "max": results_df[metric_columns].max(),
    })
 
    return summary

def comparison_row(name, results_df):
    """
    Build one summary row (mean/median across samples) for a
    configuration comparison table.
    """
    return {
        "configuration": name,
        "mean_note_precision_no_offset": results_df["note_precision_no_offset"].mean(),
        "median_note_precision_no_offset": results_df["note_precision_no_offset"].median(),
        "mean_note_recall_no_offset": results_df["note_recall_no_offset"].mean(),
        "median_note_recall_no_offset": results_df["note_recall_no_offset"].median(),
        "mean_note_f1_no_offset": results_df["note_f1_no_offset"].mean(),
        "median_note_f1_no_offset": results_df["note_f1_no_offset"].median(),
        "mean_note_f1_with_offset": results_df["note_f1"].mean(),
        "mean_predicted_notes": results_df["predicted_notes"].mean(),
        "mean_extra_notes": results_df["transcription_extra_notes"].mean(),
        "mean_missing_notes": results_df["transcription_missing_notes"].mean(),
    }
 
 
def compare_predictions(reference_notes, predictions_before, predictions_after,
                        label_before="before", label_after="after"):
    """
    General-purpose comparison helper, return a metrics table
    with a "delta" row (after minus before).
    """
    metrics_before = evaluate_note_transcription(reference_notes, predictions_before)
    output_before = analyse_transcription_output(reference_notes, predictions_before)
    row_before = {}
    row_before.update(metrics_before)
    row_before.update(output_before)
 
    metrics_after = evaluate_note_transcription(reference_notes, predictions_after)
    output_after = analyse_transcription_output(reference_notes, predictions_after)
    row_after = {}
    row_after.update(metrics_after)
    row_after.update(output_after)
 
    comparison_df = pd.DataFrame(
        [row_before, row_after],
        index=[label_before, label_after],
    )
 
    numeric_columns = comparison_df.select_dtypes(include="number").columns
    comparison_df.loc["delta", numeric_columns] = (
        comparison_df.loc[label_after, numeric_columns]
        - comparison_df.loc[label_before, numeric_columns]
    )
 
    return comparison_df
 
 
def run_case_study(sample_id, metadata_df, audio_dir, midi_dir,
                   tsv_dir, cache_dir, basic_pitch_model):
    """
    Transcribe and evaluate one sample, show its piano roll, and
    return its metrics as a Series. Uses the raw-prediction cache, so
    repeated calls for the same sample_id do not re-run the model.
    """
    sample = metadata_df[metadata_df["sample_id"] == sample_id].iloc[0]
    paths = get_sample_paths(sample, audio_dir, midi_dir, tsv_dir)
 
    reference_notes = load_ground_truth_notes(paths["tsv"])
    predicted_notes, _ = transcribe_audio_cached(
        cache_dir, sample_id, paths["audio"], basic_pitch_model,
        **BASIC_PITCH_DEFAULT_CONFIG
    )
 
    plot_transcription_piano_roll(
        reference_notes,
        predicted_notes,
        title=f"Sample {sample_id}: {sample['composer']} - {sample['piece']}",
    )
 
    metrics = evaluate_note_transcription(reference_notes, predicted_notes)
    transcription_output = analyse_transcription_output(reference_notes, predicted_notes)
 
    result = {}
    result.update(metrics)
    result.update(transcription_output)
 
    return pd.Series(result)
 
 
def run_case_study_before_after_tuning(sample_id, metadata_df, audio_dir, 
                                       midi_dir, tsv_dir, default_cache_dir,
                                       tuned_cache_dir, tuned_config, basic_pitch_model):
    """
    For one case-study sample, show:
      1. Ground truth vs TUNED config piano roll
      2. A default/tuned/delta metrics table
 
    This lets you see, sample by sample, exactly what threshold
    tuning changed -- both visually and numerically.
    """
    sample = metadata_df[metadata_df["sample_id"] == sample_id].iloc[0]
    paths = get_sample_paths(sample, audio_dir, midi_dir, tsv_dir)
    title_prefix = f"Sample {sample_id}: {sample['composer']} - {sample['piece']}"

    reference_notes = load_ground_truth_notes(paths["tsv"])
    default_notes, _ = transcribe_audio_cached(
        default_cache_dir, sample_id, paths["audio"], basic_pitch_model,
        **BASIC_PITCH_DEFAULT_CONFIG
    )
    tuned_notes, _ = transcribe_audio_cached(
        tuned_cache_dir, sample_id, paths["audio"], basic_pitch_model, **tuned_config
    )
 
    plot_transcription_piano_roll(
        reference_notes,
        tuned_notes,
        title= f"{title_prefix} (Tuned config)",
    )
 
    comparison_df = compare_predictions(
        reference_notes,
        predictions_before=default_notes,
        predictions_after=tuned_notes,
        label_before="default",
        label_after="tuned",
    )
 
    return comparison_df


from matplotlib.patches import ConnectionPatch
from matplotlib.lines import Line2D

def match_notes_for_plotting(reference_notes, predicted_notes,
                             onset_tolerance=0.05, offset_ratio=None):
    """
    Match predicted notes to PianoVAM ground-truth notes using mir_eval.

    Returns matched index pairs, missing reference-note indices,
    and extra predicted-note indices.
    """
    reference_intervals, reference_pitches = notes_to_mir_eval_arrays(
        reference_notes,
        offset_key="frame_offset",
    )
    predicted_intervals, predicted_pitches = notes_to_mir_eval_arrays(
        predicted_notes,
        offset_key="offset",
    )

    matching = mir_eval.transcription.match_notes(
        reference_intervals,
        reference_pitches,
        predicted_intervals,
        predicted_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
    )

    matched_reference_indices = {pair[0] for pair in matching}
    matched_predicted_indices = {pair[1] for pair in matching}

    missing_indices = [
        index
        for index in range(len(reference_notes))
        if index not in matched_reference_indices
    ]

    extra_indices = [
        index
        for index in range(len(predicted_notes))
        if index not in matched_predicted_indices
    ]

    return matching, missing_indices, extra_indices


def plot_transcription_alignment(reference_notes, predicted_notes, title=None,
                                 onset_tolerance=0.05, offset_ratio=None,
                                 time_range=None):
    """
    Plot predicted notes and PianoVAM ground truth in two piano-roll rows.

    Matched notes are connected by grey lines, missing reference notes
    are marked with red crosses, and extra predicted notes are marked
    with purple triangles.

    offset_ratio=None gives onset-and-pitch matching without requiring
    note offsets to match, consistent with note_f1_no_offset.
    """
    matching, missing_indices, extra_indices = match_notes_for_plotting(
        reference_notes,
        predicted_notes,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
    )

    fig, (ax_predicted, ax_reference) = plt.subplots(2, 1, 
                                                     figsize=(14, 7), 
                                                     sharex=True)

    if title is not None:
        fig.suptitle(title)

    all_pitches = (
        [note["pitch"] for note in predicted_notes]
        + [note["pitch"] for note in reference_notes]
    )

    all_offsets = (
        [note["offset"] for note in predicted_notes]
        + [note["frame_offset"] for note in reference_notes]
    )

    pitch_min = min(all_pitches) - 1
    pitch_max = max(all_pitches) + 1
    time_max = max(all_offsets)

    for note in predicted_notes:
        ax_predicted.plot(
            [note["onset"], note["offset"]],
            [note["pitch"], note["pitch"]],
            linewidth=3,
            alpha=0.8,
        )

    for note in reference_notes:
        ax_reference.plot(
            [note["onset"], note["frame_offset"]],
            [note["pitch"], note["pitch"]],
            linewidth=3,
            alpha=0.8,
        )

    def note_is_visible(note, offset_key):
        if time_range is None:
            return True

        return (
            note[offset_key] >= time_range[0]
            and note["onset"] <= time_range[1]
        )

    for reference_index, predicted_index in matching:
        reference_note = reference_notes[reference_index]
        predicted_note = predicted_notes[predicted_index]

        if not (
            note_is_visible(reference_note, "frame_offset")
            or note_is_visible(predicted_note, "offset")
        ):
            continue

        connection = ConnectionPatch(
            xyA=(predicted_note["onset"], predicted_note["pitch"]),
            coordsA=ax_predicted.transData,
            xyB=(reference_note["onset"], reference_note["pitch"]),
            coordsB=ax_reference.transData,
            color="lightgray",
            linewidth=1.8,
            zorder=10,
        )

        fig.add_artist(connection)

        ax_predicted.scatter(
            predicted_note["onset"],
            predicted_note["pitch"],
            color="lightgray",
            s=20,
            zorder=10,
        )

        ax_reference.scatter(
            reference_note["onset"],
            reference_note["pitch"],
            color="lightgray",
            s=20,
            zorder=10,
        )

    for reference_index in missing_indices:
        note = reference_notes[reference_index]
        if not note_is_visible(note, "frame_offset"):
            continue
        middle_time = note["onset"] + (note["frame_offset"] - note["onset"]) / 2
        ax_reference.scatter(middle_time, note["pitch"], marker="x", 
                             color="red", s=80, zorder=10)

    for predicted_index in extra_indices:
        note = predicted_notes[predicted_index]
        if not note_is_visible(note, "offset"):
            continue
        middle_time = note["onset"] + (note["offset"] - note["onset"]) / 2
        ax_predicted.scatter(middle_time, note["pitch"], marker="^",
                             color="purple", s=80, zorder=10)

    ax_predicted.set_title("Basic Pitch transcription")
    ax_predicted.set_ylabel("MIDI pitch")
    ax_predicted.grid(alpha=0.15)

    ax_reference.set_title("PianoVAM ground truth: audible duration including pedal")
    ax_reference.set_xlabel("Time (seconds)")
    ax_reference.set_ylabel("MIDI pitch")
    ax_reference.grid(alpha=0.15)

    if time_range is None:
        x_min = 0
        x_max = time_max
    else:
        x_min, x_max = time_range

    for ax in (ax_predicted, ax_reference):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(pitch_min, pitch_max)

    legend_elements = [
        Line2D([0], [0], color="lightgray", linewidth=1.8, label="Match"),
        Line2D([0], [0], marker="x", color="white", markeredgecolor="red",
               markersize=10, label="Missing"),
        Line2D([0], [0], marker="^", color="white", markerfacecolor="purple",
               markeredgecolor="purple", markersize=10, label="Extra")]

    ax_predicted.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    plt.show()

    return {
        "matched": len(matching),
        "missing": len(missing_indices),
        "extra": len(extra_indices),
    }