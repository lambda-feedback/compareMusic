"""
error_analysis.py
===================
False-positive / false-negative pattern analysis for AMT
transcription errors: for each extra (false positive) or missing
(false negative) note, checks a handful of simple heuristics (nearby
same-pitch fragment, harmonic-interval overlap, short duration,
position near the start, local note density) to characterise why the
error happened, not just how many there are.
"""

import mir_eval
import pandas as pd
from tqdm import tqdm

from .pianovam_loading import (
    get_sample_paths,
    load_ground_truth_notes,
    notes_to_mir_eval_arrays,
)


def notes_overlap(note_1, note_2, tolerance=0.0):
    """
    Return True if two notes overlap in time, allowing a small
    tolerance on both ends (so notes separated by a tiny gap still
    count as overlapping).
    """
    return (
        note_1["onset"] <= note_2["offset"] + tolerance
        and note_2["onset"] <= note_1["offset"] + tolerance
    )


def count_nearby_reference_notes(reference_notes, onset, onset_window_seconds):
    """
    Count reference notes starting within onset_window_seconds of a
    given onset time. Used as a simple proxy for "how many notes are
    sounding together around this moment" (chord/polyphony density).
    """
    count = 0

    for reference_note in reference_notes:
        onset_difference = abs(reference_note["onset"] - onset)
        if onset_difference <= onset_window_seconds:
            count = count + 1

    return count


def get_note_match_info(sample_id, metadata_df, audio_dir, midi_dir,
                        tsv_dir, predictions_by_sample_id, 
                        onset_tolerance=0.05):
    """
    Run mir_eval note matching once for one sample, using the
    already-computed predicted notes for that sample.
    """
    sample = metadata_df[metadata_df["sample_id"] == sample_id].iloc[0]
    paths = get_sample_paths(sample, audio_dir, midi_dir, tsv_dir)

    reference_notes = load_ground_truth_notes(paths["tsv"])
    predicted_notes = predictions_by_sample_id[str(sample_id)]

    reference_intervals, reference_pitches = notes_to_mir_eval_arrays(
        reference_notes, offset_key="frame_offset"
    )
    predicted_intervals, predicted_pitches = notes_to_mir_eval_arrays(
        predicted_notes, offset_key="offset"
    )

    # Use mir_eval to match predicted notes to reference notes, allowing a small onset tolerance.
    matching = mir_eval.transcription.match_notes(
        reference_intervals,
        reference_pitches,
        predicted_intervals,
        predicted_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=None,
    )

    # The matching is a list of (reference_index, predicted_index) pairs.
    matched_reference_indices = set(pair[0] for pair in matching)
    matched_predicted_indices = set(pair[1] for pair in matching)

    return {
        "reference_notes": reference_notes,
        "predicted_notes": predicted_notes,
        "matched_reference_indices": matched_reference_indices,
        "matched_predicted_indices": matched_predicted_indices,
    }


def summarise_fp_fn_over_train_set(train_samples, metadata_df, audio_dir,
                                   midi_dir, tsv_dir, predictions_by_sample_id,
                                   max_gap_seconds=0.05,
                                   onset_window_seconds=0.05,
                                   short_note_seconds=0.10,
):
    """
    Summarise false-positive and false-negative patterns for each
    train sample.

    FP patterns:
      - fragment_only, harmonic_interval_only, both_patterns, neither_pattern
    Extra FP flags:
      - short_extra, extra_at_beginning, dense_region_extra
    FN patterns:
      - short_missing, repeated_pitch_missing, chord_related_missing
    """
    interval_sizes = [7, 12, 19]
    rows = []

    for row_index, sample in tqdm(train_samples.iterrows(), total=len(train_samples), desc="FP/FN summary"):
        sample_id = sample["sample_id"]
        match_info = get_note_match_info(
            sample_id, metadata_df, audio_dir, midi_dir, tsv_dir, predictions_by_sample_id
        )

        reference_notes = match_info["reference_notes"]
        predicted_notes = match_info["predicted_notes"]
        matched_reference_indices = match_info["matched_reference_indices"]
        matched_predicted_indices = match_info["matched_predicted_indices"]

        # Sort predicted notes by onset time
        sorted_predicted_notes = sorted(predicted_notes, key=lambda note: note["onset"])
        # Create a list of matched predicted notes for easier pattern analysis.
        matched_predicted_notes = [predicted_notes[i] for i in matched_predicted_indices]

        if len(reference_notes) > 0:
            first_reference_onset = min(note["onset"] for note in reference_notes)
        else:
            first_reference_onset = 0.0

        total_extra = 0
        fragment_only = 0
        harmonic_interval_only = 0
        both_patterns = 0
        neither_pattern = 0
        short_extra = 0
        extra_at_beginning = 0
        dense_region_extra = 0

        for predicted_index, note in enumerate(predicted_notes):
            is_extra = predicted_index not in matched_predicted_indices

            if is_extra:
                total_extra = total_extra + 1

                has_fragment_pattern = False
                for other_note in sorted_predicted_notes:
                    same_pitch = (other_note["pitch"] == note["pitch"])
                    is_before = (other_note["offset"] <= note["onset"])
                    gap = note["onset"] - other_note["offset"]
                    # Check if there is a nearby predicted note with the same pitch that ends 
                    # before this note starts, and is within max_gap_seconds.
                    if same_pitch and is_before and (0 <= gap <= max_gap_seconds):
                        has_fragment_pattern = True

                has_harmonic_interval_pattern = False
                for matched_note in matched_predicted_notes:
                    pitch_gap = abs(note["pitch"] - matched_note["pitch"])
                    # Check if the pitch gap is a common harmonic interval (perfect fifth, octave, 
                    # or compound intervals) and if the notes overlap in time within the onset window.
                    if pitch_gap in interval_sizes and notes_overlap(note, matched_note, tolerance=onset_window_seconds):
                        has_harmonic_interval_pattern = True

                if has_fragment_pattern and has_harmonic_interval_pattern:
                    both_patterns = both_patterns + 1
                elif has_fragment_pattern:
                    fragment_only = fragment_only + 1
                elif has_harmonic_interval_pattern:
                    harmonic_interval_only = harmonic_interval_only + 1
                else:
                    neither_pattern = neither_pattern + 1

                duration = note["offset"] - note["onset"]
                # Check if the extra note is short
                if duration < short_note_seconds:
                    short_extra = short_extra + 1
                # Check if the extra note starts before the first reference note.
                if note["onset"] < first_reference_onset:
                    extra_at_beginning = extra_at_beginning + 1
                # Check if there are multiple reference notes starting near this extra note's onset.
                nearby_reference_count = count_nearby_reference_notes(
                    reference_notes, note["onset"], onset_window_seconds
                )
                if nearby_reference_count >= 2:
                    dense_region_extra = dense_region_extra + 1

        total_missing = 0
        short_missing = 0
        repeated_pitch_missing = 0
        chord_related_missing = 0

        for reference_index, note in enumerate(reference_notes):
            is_missing = reference_index not in matched_reference_indices

            if is_missing:
                total_missing = total_missing + 1

                duration = note["frame_offset"] - note["onset"]
                # Check if the missing note is short.
                if duration < short_note_seconds:
                    short_missing = short_missing + 1

                has_nearby_same_pitch_reference = False
                for other_index, other_note in enumerate(reference_notes):
                    # Check if there is another reference note with the same pitch 
                    # that starts within max_gap_seconds of this missing note's onset,
                    # mark this missing note as having a repeated pitch nearby.
                    if other_index != reference_index:
                        same_pitch = other_note["pitch"] == note["pitch"]
                        onset_gap = abs(other_note["onset"] - note["onset"])
                        if same_pitch and onset_gap <= max_gap_seconds:
                            has_nearby_same_pitch_reference = True
                if has_nearby_same_pitch_reference:
                    repeated_pitch_missing = repeated_pitch_missing + 1

                # Check if there are multiple reference notes starting near this missing note's onset, 
                # which may indicate it's part of a chord or dense region.
                nearby_reference_count = count_nearby_reference_notes(
                    reference_notes, note["onset"], onset_window_seconds
                )
                # If there are two or more reference notes starting near this missing note's onset,
                # mark it as chord-related missing.
                if nearby_reference_count >= 2:
                    chord_related_missing = chord_related_missing + 1

        rows.append({
            "sample_id": sample_id,
            "total_predicted_notes": len(predicted_notes),
            "total_reference_notes": len(reference_notes),
            "total_extra": total_extra,
            "total_missing": total_missing,
            "fragment_only": fragment_only,
            "harmonic_interval_only": harmonic_interval_only,
            "both_patterns": both_patterns,
            "neither_pattern": neither_pattern,
            "short_extra": short_extra,
            "extra_at_beginning": extra_at_beginning,
            "dense_region_extra": dense_region_extra,
            "short_missing": short_missing,
            "repeated_pitch_missing": repeated_pitch_missing,
            "chord_related_missing": chord_related_missing,
        })

    return pd.DataFrame(rows)