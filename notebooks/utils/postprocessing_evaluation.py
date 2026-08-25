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

    # Loop through each sample in the training set and evaluate the post-processed predictions
    for row_index, sample in tqdm(
        train_samples.iterrows(),
        total=len(train_samples),
        desc="Post-processing evaluation",
    ):
        sample_id = str(sample["sample_id"]) # convert sample_id to string for consistent dictionary key usage
        paths = get_sample_paths(sample, audio_dir, midi_dir, tsv_dir) # get the paths for the sample's audio, MIDI, and TSV files

        reference_notes = load_ground_truth_notes(paths["tsv"]) # load the ground truth notes from the TSV file
        predicted_notes = tuned_predictions[sample_id] # retrieve the tuned predictions for the current sample
        postprocessed_notes = postprocess_function(predicted_notes) # apply the post-processing function to the predicted notes

        metrics = evaluate_note_transcription(reference_notes, postprocessed_notes) # evaluate the post-processed notes against the ground truth notes
        transcription_output = analyse_transcription_output(reference_notes, postprocessed_notes) # analyze the transcription output for additional metrics

        # create a row dictionary to store the evaluation results for the current sample
        row = {
            "sample_id": sample_id,
            "composer": sample["composer"],
            "piece": sample["piece"],
        } 
        row.update(metrics) # add the evaluation metrics to the row dictionary
        row.update(transcription_output) # add the transcription output analysis to the row dictionary
        rows.append(row) # append the row dictionary to the list of rows for the final results DataFrame

        postprocessed_predictions[sample_id] = postprocessed_notes # store the post-processed notes in the dictionary for later use

    return pd.DataFrame(rows), postprocessed_predictions
