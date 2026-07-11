from evaluation_function.compare_MIDI import (
    compare_performance_ED,
    normalize_start_times,
    group_notes_into_events,
)

def evaluate_one_sample(sample):
    try:
        result = compare_performance_ED(sample["response"], sample["reference"])

        # Reconstruct response events (already existed)
        response_notes_norm = normalize_start_times(sample["response"]["notes"])
        response_events_norm = group_notes_into_events(response_notes_norm)

        # NEW: reconstruct reference events too, using the exact same steps
        # compare_performance_ED uses internally, so reference_index in
        # event_details lines up with this list.
        ref_notes_norm = normalize_start_times(sample["reference"]["notes"])
        ref_events_norm = group_notes_into_events(ref_notes_norm)

        if sample["response"]["notes"]:
            response_onset_offset = sample["response"]["notes"][0]["start"]
        else:
            response_onset_offset = 0.0

        return {
            "composer": sample["composer"],
            "title": sample["title"],
            "stats": result.stats,
            "event_details": result.event_details,
            "is_correct": result.is_correct,
            "response_events_normalized": response_events_norm,
            "ref_events_normalized": ref_events_norm,   # NEW
            "response_onset_offset": response_onset_offset,
            "metadata_row": sample["metadata_row"],
            "error": None,
        }
    except Exception as error:
        return {
            "composer": sample.get("composer"),
            "title": sample.get("title"),
            "stats": None,
            "event_details": None,
            "is_correct": None,
            "response_events_normalized": None,
            "ref_events_normalized": None,   # NEW
            "response_onset_offset": None,
            "metadata_row": sample.get("metadata_row"),
            "error": str(error),
        }