"""
random_search.py
===================

Random search over Basic Pitch decoding parameters (onset_threshold,
frame_threshold, minimum_note_length, melodia_trick), evaluated on
the PianoVAM train split.

The neural network inference (the expensive part) runs once per
audio file; every parameter combination is then decoded cheaply from
the same cached raw model output.
"""

import gc

import pandas as pd
from tqdm import tqdm

from evaluation_function.audio_processing import load_basic_pitch_model
from .pianovam_loading import (
    get_sample_paths,
    load_ground_truth_notes,
)
from .transcription_evaluation import (
    decode_model_output,
    evaluate_note_transcription,
    run_basic_pitch_inference,
)


def load_completed_sample_ids(progress_path):
    """
    Return sample_ids that already have results saved from a
    previous (possibly interrupted) search run.
    """
    if not progress_path.exists():
        return set()

    existing_df = pd.read_csv(progress_path)
    return set(existing_df["sample_id"].astype(str))


def run_random_search(train_samples, parameter_sets, basic_pitch_model, 
                      audio_dir, midi_dir, tsv_dir, progress_path, model_path,
                      checkpoint_every=10):
    """
    Evaluate every configuration in parameter_sets against every
    sample in train_samples, appending results to progress_path after
    each sample (so an interrupted run can resume where it left off).

    model_path is the Basic Pitch model path (e.g.
    ICASSP_2022_MODEL_PATH), used to periodically reload the model
    object and bound TensorFlow's memory growth over a long run.

    Returns basic_pitch_model, since it may be replaced with a fresh
    model object partway through the search.
    """
    completed_sample_ids = load_completed_sample_ids(progress_path)
    remaining_samples = train_samples[
        ~train_samples["sample_id"].astype(str).isin(completed_sample_ids)
    ]

    for sample_index, (row_index, sample) in enumerate(
        tqdm(remaining_samples.iterrows(), total=len(remaining_samples), desc="Random search")
    ):
        paths = get_sample_paths(sample, audio_dir, midi_dir, tsv_dir)
        reference_notes = load_ground_truth_notes(paths["tsv"])
        model_output = run_basic_pitch_inference(paths["audio"], basic_pitch_model)

        sample_rows = []
        for config_id, config in enumerate(parameter_sets):
            predicted_notes = decode_model_output(model_output, config)
            metrics = evaluate_note_transcription(reference_notes, predicted_notes)

            row = {"config_id": config_id, "sample_id": sample["sample_id"]}
            row.update(config)
            row.update(metrics)
            sample_rows.append(row)

        # Append this sample's results to disk right away, so a crash
        # later on does not lose the work already done for this sample.
        sample_df = pd.DataFrame(sample_rows)
        write_header = not progress_path.exists()
        sample_df.to_csv(progress_path, mode="a", header=write_header, index=False)

        # Release this sample's raw model output before moving on.
        del model_output
        gc.collect()

        # Periodically recreate the Basic Pitch model object.
        # TensorFlow keeps caching a new computation graph every time
        # it sees a differently-shaped input (here, every audio file
        # with a different duration), and this cache is never released
        # on its own. Recreating the Model object lets the old cached
        # graphs be garbage-collected, which prevents memory from
        # growing without bound over a long run.
        if (sample_index + 1) % checkpoint_every == 0:
            basic_pitch_model = load_basic_pitch_model(model_path)
            gc.collect()

    return basic_pitch_model


def summarise_random_search_results(progress_path):
    """
    Load the full per-sample search log and aggregate it into one
    row per configuration, sorted best-first by note_f1_no_offset.
    """
    search_details = pd.read_csv(progress_path)

    random_search_results = (
        search_details
        .groupby([
            "config_id",
            "onset_threshold",
            "frame_threshold",
            "minimum_note_length",
            "melodia_trick",
        ], as_index=False)
        .agg(
            note_precision_no_offset=("note_precision_no_offset", "mean"),
            note_recall_no_offset=("note_recall_no_offset", "mean"),
            note_f1_no_offset=("note_f1_no_offset", "mean"),
            note_f1_with_offset=("note_f1", "mean"),
            predicted_notes=("predicted_notes", "mean"),
        )
        .sort_values(
            ["note_f1_no_offset", "note_precision_no_offset", "note_recall_no_offset"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return random_search_results