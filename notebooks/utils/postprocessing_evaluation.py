"""
postprocessing_evaluation.py
===================
Evaluate post-processing steps.
"""

import pandas as pd
from tqdm import tqdm

from .pianovam_loading import (
    get_sample_paths,
    load_ground_truth_notes,
)
from .transcription_evaluation import (
    analyse_transcription_output,
    evaluate_note_transcription,
)


def evaluate_postprocessing(postprocess_function, train_samples,
                            audio_dir, midi_dir, tsv_dir, tuned_predictions):
    """
    Apply postprocess_function to each sample's tuned predictions and
    re-evaluate against ground truth.

    Returns (results_df, postprocessed_predictions).
    """
    rows = []
    postprocessed_predictions = {}

    for row_index, sample in tqdm(
        train_samples.iterrows(),
        total=len(train_samples),
        desc="Post-processing evaluation",
    ):
        sample_id = str(sample["sample_id"])
        paths = get_sample_paths(sample, audio_dir, midi_dir, tsv_dir)

        reference_notes = load_ground_truth_notes(paths["tsv"])
        predicted_notes = tuned_predictions[sample_id]
        postprocessed_notes = postprocess_function(predicted_notes)

        metrics = evaluate_note_transcription(reference_notes, postprocessed_notes)
        transcription_output = analyse_transcription_output(reference_notes, postprocessed_notes)

        row = {
            "sample_id": sample_id,
            "composer": sample["composer"],
            "piece": sample["piece"],
        }
        row.update(metrics)
        row.update(transcription_output)
        rows.append(row)

        postprocessed_predictions[sample_id] = postprocessed_notes

    return pd.DataFrame(rows), postprocessed_predictions
