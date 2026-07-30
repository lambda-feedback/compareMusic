"""
nasap_alignment_evaluation.py
=============================
Helper functions for evaluating the Phase 1 event-level MIDI alignment
pipeline on the CPJKU nASAP dataset.
"""

import csv
import os
import json

import matplotlib.pyplot as plt
import pandas as pd
import partitura as pt
import pretty_midi

from collections import Counter

from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

from evaluation_function.compare_MIDI import (
    compare_performance_ED,
    normalize_start_times,
    group_notes_into_events,
    DEFAULT_CHORD_ONSET_WINDOW,
)

ONSET_TOLERANCE = 0.05


def load_ground_truth(tsv_path):
    """
    Read a note_alignment.tsv file from the CPJKU nASAP dataset.
    Returns one dictionary per TSV row. XML and MIDI identifiers are retained
    so that individual alignment errors can be traced later.
        Keys:
            label -> 'paired', 'insertion', or 'deletion'
            xml_id -> score-note identifier, or None for insertions
            midi_id -> performance-note identifier, or None for deletions
            onset -> performance onset in seconds, or None for deletions
            pitch -> performance MIDI pitch, or None for deletions
    """
    rows = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            xml_id = row["xml_id"].strip()
            midi_id = row["midi_id"].strip()

            if midi_id == "deletion":
                rows.append({
                    "label": "deletion",
                    "xml_id": xml_id,
                    "midi_id": None,
                    "onset": None,
                    "pitch": None,
                })

            elif xml_id == "insertion":
                rows.append({
                    "label": "insertion",
                    "xml_id": None,
                    "midi_id": midi_id,
                    "onset": float(row["onset"]),
                    "pitch": int(row["pitch"]),
                })

            else:
                rows.append({
                    "label": "paired",
                    "xml_id": xml_id,
                    "midi_id": midi_id,
                    "onset": float(row["onset"]),
                    "pitch": int(row["pitch"]),
                })

    return rows


def xml_score_to_notes(xml_path):
    """
    Parse a MusicXML score into the format used by compareMusic.

    Partitura's note array provides score pitch and symbolic timing.
    Beat-based timing is preferred because it remains meaningful even when
    the MusicXML contains tempo-independent symbolic notation.
    """
    score = pt.load_score(xml_path)
    note_array = score.note_array()

    field_names = set(note_array.dtype.names or [])

    if "onset_beat" in field_names:
        onset_field = "onset_beat"
    elif "onset_quarter" in field_names:
        onset_field = "onset_quarter"
    elif "onset_div" in field_names:
        onset_field = "onset_div"
    else:
        raise KeyError(
            "Could not find an onset field in the Partitura score note array."
        )

    if "duration_beat" in field_names:
        duration_field = "duration_beat"
    elif "duration_quarter" in field_names:
        duration_field = "duration_quarter"
    elif "duration_div" in field_names:
        duration_field = "duration_div"
    else:
        raise KeyError(
            "Could not find a duration field in the Partitura score note array."
        )

    if "pitch" not in field_names:
        raise KeyError("The Partitura score note array has no 'pitch' field.")

    notes = []

    for row in note_array:
        note = {
            "pitch": int(row["pitch"]),
            "start": float(row[onset_field]),
            "duration": float(row[duration_field]),
        }

        # Retain the MusicXML note ID when Partitura supplies it.
        if "id" in field_names:
            note["xml_id"] = str(row["id"])

        notes.append(note)

    notes.sort(key=lambda note: (note["start"], note["pitch"]))
    return notes


def midi_file_to_notes(midi_path):
    """Parse a performance MIDI file into the format used by compareMusic."""
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    all_notes = []

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            all_notes.append({
                "pitch": int(note.pitch),
                # Keep full precision because rounding may change tolerance-based
                # evaluation near a matching boundary.
                "start": float(note.start),
                "duration": float(note.end - note.start),
            })

    all_notes.sort(key=lambda note: (note["start"], note["pitch"]))
    return all_notes


def build_sample(xml_path, response_path, composer, title, metadata_row):
    """
    Build one MusicXML-score / performance-MIDI sample.

    The score reference and ground-truth TSV now share the same MusicXML
    score-note representation.
    """
    score_notes = xml_score_to_notes(xml_path)
    performance_notes = midi_file_to_notes(response_path)

    if not score_notes or not performance_notes:
        return None

    return {
        "composer": composer,
        "title": title,
        "reference": {"notes": score_notes},
        "response": {"notes": performance_notes},
        "metadata_row": metadata_row,
    }


def load_samples(asap_path, composer):
    """
    Read metadata.csv and return MusicXML-score / MIDI-performance samples
    for one composer.
    """
    metadata_path = os.path.join(asap_path, "metadata.csv")
    samples = []

    with open(metadata_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if row.get("composer", "").strip() != composer:
                continue

            xml_path = os.path.join(asap_path, row.get("xml_score", "").strip())
            response_path = os.path.join(asap_path, row.get("midi_performance", "").strip())
            tsv_path = os.path.join(asap_path, row.get("note_alignments", "").strip())
            
            if os.path.isfile(xml_path) and os.path.isfile(response_path) and os.path.isfile(tsv_path):
                sample = build_sample(
                xml_path,
                response_path,
                row.get("composer", "Unknown"),
                row.get("title", "Unknown"),
                dict(row),
            )
                if sample is not None:
                    samples.append(sample)

    return samples


def run_alignment_on_samples(samples):
    """Run compare_performance_ED for every loaded sample."""
    results = []

    for sample_index, sample in enumerate(samples):
        result = compare_performance_ED(
            sample["response"],
            sample["reference"],
        )

        response_notes_normalized = normalize_start_times(
            sample["response"]["notes"]
        )
        response_events_normalized = group_notes_into_events(
            response_notes_normalized
        )

        ref_notes_normalized = normalize_start_times(
            sample["reference"]["notes"]
        )
        ref_events_normalized = group_notes_into_events(
            ref_notes_normalized
        )

        response_onset_offset = float(
            sample["response"]["notes"][0]["start"]
        )

        results.append({
            "sample_index": sample_index,
            "composer": sample["composer"],
            "title": sample["title"],
            "stats": result.stats,
            "event_details": result.event_details,
            "is_correct": result.is_correct,
            "response_events_normalized": response_events_normalized,
            "ref_events_normalized": ref_events_normalized,
            "response_onset_offset": response_onset_offset,
        })

    return results


def convert_pipeline_output(event_details, response_events, ref_events):
    """
    Convert event-level alignment output into note-level alignment labels.
    Alignment labels describe note correspondence, not pitch correctness.

    Rules:
        missing:
            Every note in the reference event becomes a deletion.
        extra:
            Every note in the response event becomes an insertion.
        match/replacement:
            Pair as many response and reference notes as possible.
            Different pitches may still form a paired alignment.
            Remaining response notes become insertions.
            Remaining reference notes become deletions.

    This implementation guarantees:
        paired + insertion == total response notes
        paired + deletion == total reference notes
    """
    predictions = []

    for event in event_details:
        operation = event["operation_type"]

        response_event = None
        reference_event = None

        if event.get("response_index") is not None:
            response_index = event["response_index"] - 1
            if (response_index < 0) or (response_index >= len(response_events)):
                raise IndexError("response_index is outside response_events")
            response_event = response_events[response_index]

        if event.get("reference_index") is not None:
            reference_index = event["reference_index"] - 1
            if (reference_index < 0) or (reference_index >= len(ref_events)):
                raise IndexError("reference_index is outside ref_events")
            reference_event = ref_events[reference_index]

        # for missing event
        if operation == "missing":
            for reference_note in reference_event["notes"]:
                predictions.append({
                    "onset": None,
                    "pitch": None,
                    "label": "deletion",
                })
        # for extra event
        elif operation == "extra":
            for response_note in response_event["notes"]:
                predictions.append({
                    "onset": float(response_note["start"]),
                    "pitch": int(response_note["pitch"]),
                    "label": "insertion",
                })
        # for matched or replacement event
        else:
            response_notes = sorted(
                response_event["notes"],
                key=lambda note: int(note["pitch"])
            )
            reference_notes = sorted(
                reference_event["notes"],
                key=lambda note: int(note["pitch"])
            )
            pair_count = min(len(response_notes), len(reference_notes))

            # A pitch difference does not change the alignment label:
            # it remains paired.
            for index in range(pair_count):
                response_note = response_notes[index]
                predictions.append({
                    "onset": float(response_note["start"]),
                    "pitch": int(response_note["pitch"]),
                    "label": "paired",
                })

            # Response side contains more notes.
            for response_note in response_notes[pair_count:]:
                predictions.append({
                    "onset": float(response_note["start"]),
                    "pitch": int(response_note["pitch"]),
                    "label": "insertion",
                })

            # Reference side contains more notes.
            missing_count = len(reference_notes) - pair_count
            for i in range(missing_count):
                predictions.append({
                    "onset": None,
                    "pitch": None,
                    "label": "deletion",
                })

    return predictions


def count_onset_matches(gt_onsets, predicted_onsets, tolerance):
    """Count one-to-one onset matches while preserving chord notes."""
    gt_onsets = sorted(gt_onsets)
    predicted_onsets = sorted(predicted_onsets)

    gt_index = 0
    predicted_index = 0
    matches = 0

    while gt_index < len(gt_onsets) and predicted_index < len(predicted_onsets):
        difference = predicted_onsets[predicted_index] - gt_onsets[gt_index]

        if abs(difference) <= tolerance:
            matches += 1
            gt_index += 1
            predicted_index += 1
        elif difference < -tolerance:
            predicted_index += 1
        else:
            gt_index += 1

    return matches


def compute_metrics(ground_truth, predictions, onset_offset=0.0):
    """Compute note-level precision, recall, and F1."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # 1. Paired notes: compare onset only.
    gt_paired = [
        row["onset"]
        for row in ground_truth
        if row["label"] == "paired" and row["onset"] is not None
    ]
    predicted_paired = [
        row["onset"] + onset_offset
        for row in predictions
        if row["label"] == "paired" and row["onset"] is not None
    ]

    paired_tp = count_onset_matches(
        gt_paired, predicted_paired, ONSET_TOLERANCE
    )
    total_tp += paired_tp
    total_fp += len(predicted_paired) - paired_tp
    total_fn += len(gt_paired) - paired_tp

    # 2. Insertions: compare both pitch and onset.
    gt_insertions = {}
    predicted_insertions = {}

    for row in ground_truth:
        if (
            row["label"] == "insertion"
            and row["pitch"] is not None
            and row["onset"] is not None
        ):
            pitch = int(row["pitch"])
            if pitch not in gt_insertions:
                gt_insertions[pitch] = []
            gt_insertions[pitch].append(float(row["onset"]))

    for row in predictions:
        if (
            row["label"] == "insertion"
            and row["pitch"] is not None
            and row["onset"] is not None
        ):
            pitch = int(row["pitch"])
            if pitch not in predicted_insertions:
                predicted_insertions[pitch] = []
            predicted_insertions[pitch].append(
                float(row["onset"]) + onset_offset
            )

    for pitch in set(gt_insertions) | set(predicted_insertions):
        gt_onsets = gt_insertions.get(pitch, [])
        predicted_onsets = predicted_insertions.get(pitch, [])

        insertion_tp = count_onset_matches(
            gt_onsets,
            predicted_onsets,
            ONSET_TOLERANCE,
        )
        total_tp += insertion_tp
        total_fp += len(predicted_onsets) - insertion_tp
        total_fn += len(gt_onsets) - insertion_tp

    # 3. Deletions: count-based because the pipeline does not return xml_id.
    gt_deletions = sum(
        row["label"] == "deletion" for row in ground_truth
    )
    predicted_deletions = sum(
        row["label"] == "deletion" for row in predictions
    )

    deletion_tp = min(gt_deletions, predicted_deletions)
    total_tp += deletion_tp
    total_fp += predicted_deletions - deletion_tp
    total_fn += gt_deletions - deletion_tp

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0 else 0.0
    )
    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0 else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def get_note_centre(note):
    return (
        note["start"] + note["duration"] / 2,
        note["pitch"],
    )


def event_overlaps_range(event, start, end):
    for note in event["notes"]:
        note_start = note["start"]
        note_end = note["start"] + note["duration"]
        if note_end >= start and note_start <= end:
            return True
    return False


def pair_event_notes(ref_event, response_event):
    """
    Pair notes inside two aligned events for plotting.

    Rules:
        1. Pair notes with the same pitch first.
        2. Pair remaining notes in ascending pitch order.
        3. If the events contain different numbers of notes, only pair
           as many notes as possible.
    """
    ref_notes = sorted(
        ref_event["notes"],
        key=lambda note: (note["pitch"], note["start"]),
    )

    response_notes = sorted(
        response_event["notes"],
        key=lambda note: (note["pitch"], note["start"]),
    )

    note_pairs = []

    used_ref_indices = set()
    used_response_indices = set()

    # First pair notes with identical pitches.
    for response_index, response_note in enumerate(response_notes):
        matching_ref_index = None

        for ref_index, ref_note in enumerate(ref_notes):
            ref_is_available = ref_index not in used_ref_indices
            pitches_match = ref_note["pitch"] == response_note["pitch"]

            if ref_is_available and pitches_match:
                matching_ref_index = ref_index

        if matching_ref_index is not None:
            note_pairs.append(
                (
                    ref_notes[matching_ref_index],
                    response_note,
                )
            )

            used_ref_indices.add(matching_ref_index)
            used_response_indices.add(response_index)

    # Collect notes that were not paired by pitch.
    remaining_ref_notes = []

    for ref_index, ref_note in enumerate(ref_notes):
        if ref_index not in used_ref_indices:
            remaining_ref_notes.append(ref_note)

    remaining_response_notes = []

    for response_index, response_note in enumerate(response_notes):
        if response_index not in used_response_indices:
            remaining_response_notes.append(response_note)

    # Pair the remaining notes by ascending pitch.
    remaining_pair_count = min(
        len(remaining_ref_notes),
        len(remaining_response_notes),
    )

    for pair_index in range(remaining_pair_count):
        note_pairs.append(
            (
                remaining_ref_notes[pair_index],
                remaining_response_notes[pair_index],
            )
        )

    return note_pairs


def plot_alignment(ref_events, response_events, event_details, ref_start, ref_end):
    reference_positions = []

    for position, detail in enumerate(event_details):
        reference_index = detail.get("reference_index")

        if reference_index is not None:
            ref_event = ref_events[reference_index - 1]

            if event_overlaps_range(
                ref_event,
                ref_start,
                ref_end,
            ):
                reference_positions.append(position)

    if not reference_positions:
        raise ValueError(
            "No reference events were found in this range."
        )

    first_position = min(reference_positions)
    last_position = max(reference_positions)

    selected_details = []

    for position in range(first_position, last_position + 1):
        detail = event_details[position]
        operation = detail["operation_type"]
        reference_index = detail.get("reference_index")

        include_detail = operation == "extra"

        if reference_index is not None:
            ref_event = ref_events[reference_index - 1]

            if event_overlaps_range(ref_event, ref_start, ref_end):
                include_detail = True

        if include_detail:
            selected_details.append(detail)

    response_starts = []
    response_ends = []

    for detail in selected_details:
        response_index = detail.get("response_index")

        if response_index is not None:
            response_event = response_events[response_index - 1]

            for note in response_event["notes"]:
                response_starts.append(note["start"])
                response_ends.append(note["start"] + note["duration"])

    if response_starts:
        response_start = min(response_starts) - 0.5
        response_end = max(response_ends) + 0.5
    else:
        response_start = ref_start
        response_end = ref_end

    visible_response_events = []
    for event in response_events:
        if event_overlaps_range(event, response_start, response_end):
            visible_response_events.append(event)

    visible_ref_events = []
    for event in ref_events:
        if event_overlaps_range(event, ref_start, ref_end):
            visible_ref_events.append(event)

    visible_pitches = []
    for event in visible_response_events:
        for note in event["notes"]:
            visible_pitches.append(note["pitch"])
    for event in visible_ref_events:
        for note in event["notes"]:
            visible_pitches.append(note["pitch"])

    fig, (ax_response, ax_ref) = plt.subplots(2, 1,
        figsize=(15, 8), height_ratios=[1, 1])

    note_colour = "tab:blue"
    match_colour = "0.65"
    replacement_colour = "tab:orange"
    missing_colour = "tab:red"
    extra_colour = "tab:purple"

    # Draw performance piano roll.
    for event in visible_response_events:
        for note in event["notes"]:
            ax_response.hlines(
                y=note["pitch"],
                xmin=note["start"],
                xmax=note["start"] + note["duration"],
                linewidth=5,
                color=note_colour,
                zorder=3,
            )

    ax_response.set_xlim(response_start, response_end)
    ax_response.set_ylabel("Performance MIDI pitch")
    ax_response.set_title("Performance")

    # Draw reference piano roll.
    for event in visible_ref_events:
        for note in event["notes"]:
            ax_ref.hlines(
                y=note["pitch"],
                xmin=note["start"],
                xmax=note["start"] + note["duration"],
                linewidth=5,
                color=note_colour,
                zorder=3,
            )

    ax_ref.set_xlim(ref_start, ref_end)
    ax_ref.set_xlabel("Normalised time (seconds)")
    ax_ref.set_ylabel("Reference MIDI pitch")
    ax_ref.set_title("Reference")

    if visible_pitches:
        pitch_min = min(visible_pitches) - 1
        pitch_max = max(visible_pitches) + 1
        ax_response.set_ylim(pitch_min, pitch_max)
        ax_ref.set_ylim(pitch_min, pitch_max)

    connector_count = 0

    # Draw one connector for every paired note.
    for detail in selected_details:
        operation = detail["operation_type"]
        reference_index = detail.get("reference_index")
        response_index = detail.get("response_index")

        both_events_exist = (
            reference_index is not None
            and response_index is not None
        )

        if both_events_exist:
            ref_event = ref_events[reference_index - 1]
            response_event = response_events[response_index - 1]

            note_pairs = pair_event_notes(ref_event, response_event)

            if operation == "match":
                line_style = "-"
                line_width = 0.8
                line_alpha = 0.35
                line_colour = match_colour
            else:
                line_style = "--"
                line_width = 2.2
                line_alpha = 0.9
                line_colour = replacement_colour

            for ref_note, response_note in note_pairs:
                ref_x, ref_y = get_note_centre(ref_note)

                response_x, response_y = get_note_centre(response_note)

                response_visible = (
                    response_start
                    <= response_x
                    <= response_end
                )

                reference_visible = (
                    ref_start
                    <= ref_x
                    <= ref_end
                )

                if response_visible and reference_visible:
                    connector = ConnectionPatch(
                        xyA=(response_x, response_y),
                        coordsA=ax_response.transData,
                        xyB=(ref_x, ref_y),
                        coordsB=ax_ref.transData,
                        linestyle=line_style,
                        linewidth=line_width,
                        color=line_colour,
                        alpha=line_alpha,
                        clip_on=False,
                        zorder=2,
                    )

                    fig.add_artist(connector)
                    connector_count += 1

    missing_count = 0
    extra_count = 0

    for detail in selected_details:
        operation = detail["operation_type"]

        if operation == "missing":
            reference_index = detail.get("reference_index")

            if reference_index is not None:
                ref_event = ref_events[reference_index - 1]

                for note in ref_event["notes"]:
                    marker_x, marker_y = get_note_centre(note)

                    if ref_start <= marker_x <= ref_end:
                        ax_ref.scatter(
                            marker_x,
                            marker_y,
                            marker="x",
                            s=130,
                            linewidths=3,
                            color=missing_colour,
                            zorder=10,
                        )

                        missing_count += 1

        elif operation == "extra":
            response_index = detail.get("response_index")

            if response_index is not None:
                response_event = response_events[
                    response_index - 1
                ]

                for note in response_event["notes"]:
                    marker_x, marker_y = get_note_centre(note)

                    if (
                        response_start
                        <= marker_x
                        <= response_end
                    ):
                        ax_response.scatter(
                            marker_x,
                            marker_y,
                            marker="^",
                            s=110,
                            color=extra_colour,
                            edgecolors="black",
                            linewidths=1,
                            zorder=10,
                        )

                        extra_count += 1

    legend_items = [
        Line2D([0], [0], color=note_colour, linewidth=5, label="MIDI note"),
        Line2D([0], [0], color=match_colour, linestyle="-", linewidth=1.2, label="Matched note"),
        Line2D([0], [0], color=replacement_colour, linestyle="--", 
               linewidth=2.2, label="Replacement note"),
        Line2D([0], [0], color=missing_colour, marker="x", linestyle="None", 
               markersize=9, markeredgewidth=2.5, label="Missing note"),
        Line2D([0], [0], color=extra_colour, marker="^", markeredgecolor="black",
               linestyle="None", markersize=9, label="Extra note"),
    ]

    ax_response.legend(handles=legend_items, loc="upper right", framealpha=0.9)
    ax_response.grid(axis="x", alpha=0.25)
    ax_ref.grid(axis="x", alpha=0.25)
    fig.suptitle("Note-level visualisation of event-level alignment")
    plt.tight_layout()
    plt.show()

    print("Selected operations:", len(selected_details))
    print("Note connectors:", connector_count)
    print("Missing note markers:", missing_count)
    print("Extra note markers:", extra_count)


def plot_case(ax_response, ax_ref, fig, ref_events, response_events, event_details, ref_start, ref_end, title):
    selected_details = []

    for detail in event_details:
        reference_index = detail.get("reference_index")

        if reference_index is not None:
            ref_event = ref_events[reference_index - 1]

            if event_overlaps_range(ref_event, ref_start, ref_end):
                selected_details.append(detail)

    selected_positions = []

    for position, detail in enumerate(event_details):
        if detail in selected_details:
            selected_positions.append(position)

    if selected_positions:
        first_position = min(selected_positions)
        last_position = max(selected_positions)

        for position in range(first_position, last_position + 1):
            detail = event_details[position]

            if detail["operation_type"] == "extra" and detail not in selected_details:
                selected_details.append(detail)

    response_starts = []
    response_ends = []

    for detail in selected_details:
        response_index = detail.get("response_index")

        if response_index is not None:
            response_event = response_events[response_index - 1]

            for note in response_event["notes"]:
                response_starts.append(note["start"])
                response_ends.append(note["start"] + note["duration"])

    if response_starts:
        response_start = min(response_starts) - 0.2
        response_end = max(response_ends) + 0.2
    else:
        response_start = ref_start
        response_end = ref_end

    visible_ref_events = []
    visible_response_events = []
    visible_pitches = []

    for event in ref_events:
        if event_overlaps_range(event, ref_start, ref_end):
            visible_ref_events.append(event)

    for event in response_events:
        if event_overlaps_range(event, response_start, response_end):
            visible_response_events.append(event)

    for event in visible_ref_events:
        for note in event["notes"]:
            visible_pitches.append(note["pitch"])

    for event in visible_response_events:
        for note in event["notes"]:
            visible_pitches.append(note["pitch"])

    note_colour = "tab:blue"
    match_colour = "0.65"
    replacement_colour = "tab:orange"
    missing_colour = "tab:red"
    extra_colour = "tab:purple"

    for event in visible_response_events:
        for note in event["notes"]:
            ax_response.hlines(note["pitch"], note["start"], note["start"] + note["duration"], linewidth=5, color=note_colour)

    for event in visible_ref_events:
        for note in event["notes"]:
            ax_ref.hlines(note["pitch"], note["start"], note["start"] + note["duration"], linewidth=5, color=note_colour)

    ax_response.set_xlim(response_start, response_end)
    ax_ref.set_xlim(ref_start, ref_end)

    if visible_pitches:
        pitch_min = min(visible_pitches) - 1
        pitch_max = max(visible_pitches) + 1
        ax_response.set_ylim(pitch_min, pitch_max)
        ax_ref.set_ylim(pitch_min, pitch_max)

    for detail in selected_details:
        operation = detail["operation_type"]
        reference_index = detail.get("reference_index")
        response_index = detail.get("response_index")

        if reference_index is not None and response_index is not None:
            ref_event = ref_events[reference_index - 1]
            response_event = response_events[response_index - 1]
            note_pairs = pair_event_notes(ref_event, response_event)

            if operation == "match":
                line_colour = match_colour
                line_style = "-"
                line_width = 0.8
                line_alpha = 0.4
            else:
                line_colour = replacement_colour
                line_style = "--"
                line_width = 2
                line_alpha = 0.9

            for ref_note, response_note in note_pairs:
                ref_x, ref_y = get_note_centre(ref_note)
                response_x, response_y = get_note_centre(response_note)

                connector = ConnectionPatch(
                    xyA=(response_x, response_y),
                    coordsA=ax_response.transData,
                    xyB=(ref_x, ref_y),
                    coordsB=ax_ref.transData,
                    color=line_colour,
                    linestyle=line_style,
                    linewidth=line_width,
                    alpha=line_alpha,
                    clip_on=False
                )

                fig.add_artist(connector)

        if operation == "missing" and reference_index is not None:
            ref_event = ref_events[reference_index - 1]

            for note in ref_event["notes"]:
                x, y = get_note_centre(note)
                ax_ref.scatter(x, y, marker="x", s=120, linewidths=3, color=missing_colour, zorder=10)

        if operation == "extra" and response_index is not None:
            response_event = response_events[response_index - 1]

            for note in response_event["notes"]:
                x, y = get_note_centre(note)
                ax_response.scatter(x, y, marker="^", s=100, color=extra_colour, edgecolors="black", zorder=10)

    ax_response.set_title(title)
    ax_response.grid(axis="x", alpha=0.2)
    ax_ref.grid(axis="x", alpha=0.2)
    ax_ref.set_xlabel("Time (s)")


def evaluate_alignment_dataset(
    asap_path,
    composer="Bach",
):
    """
    Load one composer's nASAP samples, run the Phase 1 pipeline and evaluate
    the predicted note-level alignment labels against the nASAP ground truth.

    Returns:
        samples:
            Loaded score/performance pairs.

        all_results:
            Raw outputs from compare_performance_ED and normalised events.

        evaluation_df:
            One row per evaluated performance.
    """
    samples = load_samples(asap_path, composer)
    all_results = run_alignment_on_samples(samples)

    evaluation_rows = []

    for sample, result in zip(samples, all_results):
        metadata_row = sample["metadata_row"]
        tsv_relative_path = (
            metadata_row.get("note_alignments") or ""
        ).strip()
        tsv_path = os.path.join(asap_path, tsv_relative_path)

        if not os.path.isfile(tsv_path):
            print("TSV not found:", tsv_path)
            continue

        ground_truth = load_ground_truth(tsv_path)

        predictions = convert_pipeline_output(
            result["event_details"],
            result["response_events_normalized"],
            result["ref_events_normalized"],
        )

        metrics = compute_metrics(
            ground_truth,
            predictions,
            onset_offset=result["response_onset_offset"],
        )

        evaluation_rows.append({
            "sample_index": result["sample_index"],
            "composer": result["composer"],
            "title": result["title"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "ground_truth_count": len(ground_truth),
            "prediction_count": len(predictions),
        })

    evaluation_df = pd.DataFrame(evaluation_rows)

    return samples, all_results, evaluation_df


def prepare_alignment_case_study(samples, all_results, sample_index, asap_path=None):
    """
    Collect the data needed for one nASAP alignment case study.

    When asap_path is provided, the function also:

    - loads the nASAP ground-truth alignment;
    - converts the pipeline output into note-level labels;
    - restores the original response onset offset;
    - computes evaluation metrics;
    - counts paired, insertion and deletion labels;
    - checks whether every response and reference note is represented
      exactly once in the converted pipeline output.

    Returns:
        A dictionary containing the selected sample, pipeline result,
        events, labels, metrics, label counts and conservation checks.
    """
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(
            f"sample_index must be between 0 and {len(samples) - 1}."
        )

    sample = samples[sample_index]
    result = all_results[sample_index]

    ref_events = result["ref_events_normalized"]
    response_events = result["response_events_normalized"]
    event_details = result["event_details"]

    case_study = {
        "sample_index": sample_index,
        "sample": sample,
        "result": result,
        "ref_events": ref_events,
        "response_events": response_events,
        "event_details": event_details,
    }

    if asap_path is None:
        return case_study

    metadata_row = sample["metadata_row"]

    paths = {
        "tsv": os.path.join(
            asap_path,
            metadata_row["note_alignments"].strip(),
        ),
        "xml": os.path.join(
            asap_path,
            metadata_row["xml_score"].strip(),
        ),
        "midi_score": os.path.join(
            asap_path,
            metadata_row["midi_score"].strip(),
        ),
        "midi_performance": os.path.join(
            asap_path,
            metadata_row["midi_performance"].strip(),
        ),
    }

    ground_truth = load_ground_truth(paths["tsv"])

    predictions = convert_pipeline_output(
        event_details,
        response_events,
        ref_events,
    )

    # convert_pipeline_output uses the normalised response event times.
    # Restore the original MIDI onset times before comparing with nASAP.
    response_onset_offset = result["response_onset_offset"]

    for prediction in predictions:
        if prediction["onset"] is not None:
            prediction["onset"] += response_onset_offset

    metrics = compute_metrics(ground_truth, predictions)

    ground_truth_counts = Counter(note["label"] for note in ground_truth)

    prediction_counts = Counter(note["label"] for note in predictions)

    ground_truth_response_note_count = (
        ground_truth_counts["paired"]
        + ground_truth_counts["insertion"]
    )

    ground_truth_reference_note_count = (
        ground_truth_counts["paired"]
        + ground_truth_counts["deletion"]
    )

    prediction_response_note_count = (
        prediction_counts["paired"]
        + prediction_counts["insertion"]
    )

    prediction_reference_note_count = (
        prediction_counts["paired"]
        + prediction_counts["deletion"]
    )

    actual_response_note_count = sum(
        len(event["notes"])
        for event in response_events
    )

    actual_reference_note_count = sum(
        len(event["notes"])
        for event in ref_events
    )

    response_notes_conserved = (
        prediction_response_note_count
        == actual_response_note_count
    )

    reference_notes_conserved = (
        prediction_reference_note_count
        == actual_reference_note_count
    )

    case_study.update({
        "paths": paths,
        "ground_truth": ground_truth,
        "predictions": predictions,
        "metrics": metrics,
        "ground_truth_counts": ground_truth_counts,
        "prediction_counts": prediction_counts,
        "ground_truth_response_note_count": ground_truth_response_note_count,
        "ground_truth_reference_note_count": ground_truth_reference_note_count,
        "prediction_response_note_count": prediction_response_note_count,
        "prediction_reference_note_count": prediction_reference_note_count,
        "actual_response_note_count": actual_response_note_count,
        "actual_reference_note_count": actual_reference_note_count,
        "response_notes_conserved": response_notes_conserved,
        "reference_notes_conserved": reference_notes_conserved,
    })

    return case_study

def print_alignment_case_summary(case_study):
    """
    Print the label counts and conservation checks for one case study.
    """
    result = case_study["result"]
    ground_truth_counts = case_study["ground_truth_counts"]
    prediction_counts = case_study["prediction_counts"]

    print("Piece:", result["composer"], result["title"])
    print("Ground-truth labels:", len(case_study["ground_truth"]))
    print("Pipeline labels:", len(case_study["predictions"]))

    print("\n=== Ground truth ===")
    print("Paired:", ground_truth_counts["paired"])
    print("Insertion:", ground_truth_counts["insertion"])
    print("Deletion:", ground_truth_counts["deletion"])
    print("Performance notes represented:", case_study["ground_truth_response_note_count"])
    print("Reference notes represented:", case_study["ground_truth_reference_note_count"])

    print("\n=== Pipeline predictions ===")
    print("Paired:", prediction_counts["paired"])
    print("Insertion:", prediction_counts["insertion"])
    print("Deletion:", prediction_counts["deletion"])
    print("Performance notes represented:", case_study["prediction_response_note_count"])
    print("Reference notes represented:", case_study["prediction_reference_note_count"])

    print("\n=== Original event data ===")
    print("Notes in response_events:", case_study["actual_response_note_count"])
    print("Notes in ref_events:", case_study["actual_reference_note_count"])

    print("\n=== Conservation checks ===")
    print("Every response note represented exactly once:", case_study["response_notes_conserved"])
    print("Every reference note represented exactly once:", case_study["reference_notes_conserved"])


def load_typical_midi_test_case(
    json_path,
    case_name,
    chord_onset_window=DEFAULT_CHORD_ONSET_WINDOW,
):
    """
    Load one named JSON MIDI test case and prepare its alignment data.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        test_data = json.load(file)

    selected_case = None

    for case in test_data["test_cases"]:
        if case["name"] == case_name:
            selected_case = case
            break

    if selected_case is None:
        raise ValueError(
            f"Test case '{case_name}' was not found in {json_path}."
        )

    result = compare_performance_ED(
        selected_case["response"],
        selected_case["reference"],
        chord_onset_window=chord_onset_window,
    )

    reference_notes = normalize_start_times(
        selected_case["reference"]["notes"]
    )
    response_notes = normalize_start_times(
        selected_case["response"]["notes"]
    )

    reference_events = group_notes_into_events(
        reference_notes,
        chord_onset_window=chord_onset_window,
    )
    response_events = group_notes_into_events(
        response_notes,
        chord_onset_window=chord_onset_window,
    )

    return {
        "case": selected_case,
        "result": result,
        "reference_events": reference_events,
        "response_events": response_events,
        "event_details": result.event_details,
    }


def plot_typical_alignment_cases(
    reference_events,
    response_events,
    event_details,
    cases,
    figure_size=(18, 6),
    title="Typical note alignment cases",
):
    """
    Plot several selected alignment ranges in one two-row figure.

    Args:
        cases:
            Sequence of tuples:
                (case_title, reference_start, reference_end)
    """
    fig, axes = plt.subplots(
        2,
        len(cases),
        figsize=figure_size,
        squeeze=False,
    )

    for column, case in enumerate(cases):
        case_title, ref_start, ref_end = case

        plot_case(
            axes[0, column],
            axes[1, column],
            fig,
            reference_events,
            response_events,
            event_details,
            ref_start,
            ref_end,
            case_title,
        )

    axes[0, 0].set_ylabel("Performance\nMIDI pitch")
    axes[1, 0].set_ylabel("Reference\nMIDI pitch")

    legend_items = [
    Line2D([0], [0], color="tab:blue", linewidth=5, label="MIDI note"),
    Line2D([0], [0], color="0.65", linewidth=1.2, label="Match"),
    Line2D([0], [0], color="tab:orange", linestyle="--", linewidth=2, label="Replacement"),
    Line2D([0], [0], color="tab:red", marker="x", linestyle="None", markersize=9, label="Missing"),
    Line2D([0], [0], color="tab:purple", marker="^", markeredgecolor="black", linestyle="None", markersize=9, label="Extra")
    ]

    fig.legend(handles=legend_items, loc="lower center", ncol=5)
    fig.suptitle(title, fontsize=15)
    plt.tight_layout(rect=[0, 0.1, 1, 0.93])
    plt.show()
