"""
compare_MIDI.py
================
Core MIDI evaluation pipeline for the compareMusic evaluation function.

Pipeline overview (called in order by compare_performance_ED):
    Step 0 -- normalize_start_times     (make first note start at t = 0.0)
              group_notes_into_events   (group simultaneous notes into chords)
    Step 1 -- event_alignment_ED        (edit-distance alignment)
    Step 2 -- estimate_global_timing    (linear regression for tempo drift)
              estimate_global_duration_scale
    Step 3 -- event_level_feedback      (note/chord-level feedback)
    Step 4 -- compute_stats             (summary counts)
    Step 5 -- generate_feedback_message (human-readable text)
              polished_feedback_message (polished version of human-readable text)
    Step 6 -- is_correct                (overall pass/fail judgement)
"""


import numpy as np
from collections import Counter

# Default thresholds / parameters
# Teachers can override any of these via the params dict in evaluation_function.
# ------------------------------------------------------------------------------
# Gap penalty: cost of leaving a note unaligned (insertion/deletion)
DEFAULT_GAP_PENALTY = 6

# Timing: |response_start - predicted_start| / Inter-Onset Interval (IOI) must be below this.
# e.g. 0.20 means the start can be off by up to 20% of the inter-onset interval.
TIMING_RELATIVE_THRESHOLD = 0.20

# Duration: |response_dur / ref_dur - 1| must be below this.
# e.g. 0.25 means the student's duration can be off by up to 25% of the reference.
DURATION_RELATIVE_THRESHOLD = 0.25

# Thresholds that trigger a global tempo comment in the overview.
GLOBAL_SLOW_THRESHOLD = 1.15   # timing_scale > 1.15  -> "overall too slow"
GLOBAL_FAST_THRESHOLD = 0.85   # timing_scale < 0.85  -> "overall too fast"

# Default threshold: notes starting within 50ms are grouped as one chord.
DEFAULT_CHORD_ONSET_WINDOW = 0.05

# template and helper functions for chords
# ------------------------------------------------------------------------------
# Chord template dictionary.
# Each entry maps a chord quality name to a set of pitch class intervals,
# where the root note is normalised to 0.
# Only the 4 triad types are included, following Muller (2021) Chapter 5.
CHORD_TEMPLATES = {
    "major": set([0, 4, 7]),
    "minor": set([0, 3, 7]),
    "diminished": set([0, 3, 6]),
    "augmented": set([0, 4, 8]),
}
 
# Pitch class names for human-readable feedback messages.
PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F",
                     "F#", "G", "G#", "A", "A#", "B"]

# Chord helper functions
def get_pitch_class_set(notes):
    """
    Convert a list of notes to a set of pitch classes (each pitch mod 12).

    Args:
        notes: list of note dicts, each with a "pitch" key.

    Returns:
        set of ints, each in range [0, 11].
    """
    return set(note["pitch"] % 12 for note in notes)
 
def identify_chord_name(notes):
    """
    Identify the chord name (e.g. "C major", "A minor") from a list of notes
    by matching their pitch class set against CHORD_TEMPLATES.
    If no match is found, returns "unknown chord".
 
    Args:
        notes: list of note dicts, each with a "pitch" key.
 
    Returns:
        str: chord name e.g. "C major", or "unknown chord".
    """
    pitch_classes = get_pitch_class_set(notes)
 
    for root_pc in pitch_classes:
        normalised = set((pc - root_pc) % 12 for pc in pitch_classes)
        for chord_type, template in CHORD_TEMPLATES.items():
            if normalised == template:
                root_name = PITCH_CLASS_NAMES[root_pc]
                return root_name + " " + chord_type
 
    return "unknown chord"
 
def compute_chord_accuracy(ref_notes, res_notes):
    """
    Compute the chord accuracy score A (modified Devaney's method), but
    use the actual MIDI pitch (with octave) instead of pitch class, 
    so that the score is sensitive to octave errors.: 
    A = (C - I + |y|) / (2 * |y|)
    where:
        C = number of correctly matched notes
        I = number of extra notes
        |y| = number of notes in the reference chord
    A = 1.0 means perfectly correct. 
    A = 0.0 means nothing correct and many unexpected notes are played.
 
    Args:
        ref_notes: list of note dicts for the reference chord.
        res_notes: list of note dicts for the response chord.
 
    Returns:
        accuracy: float in [0, 1]
        correct_pitches: sorted list of matched MIDI pitches
        missing_pitches: sorted list of missing MIDI pitches
        extra_pitches: sorted list of extra MIDI pitches
    """
    ref_counts = Counter(note["pitch"] for note in ref_notes)
    res_counts = Counter(note["pitch"] for note in res_notes)

    correct_pitches = []
    missing_pitches = []
    extra_pitches = []

    # For every distinct pitch on either side, match up copies one-to-one.
    # Any leftover ref copies are missing; any leftover res copies are extra.
    all_pitches = sorted(set(ref_counts) | set(res_counts))
    for pitch in all_pitches:
        matched = min(ref_counts[pitch], res_counts[pitch])
        correct_pitches.extend([pitch] * matched)
        if ref_counts[pitch] > matched:
            missing_pitches.extend(
                [pitch] * (ref_counts[pitch] - matched)
            )
        if res_counts[pitch] > matched:
            extra_pitches.extend(
                [pitch] * (res_counts[pitch] - matched)
            )

    C = len(correct_pitches)
    I = len(extra_pitches)
    ref_size = len(ref_notes)   # total note count, repeats included

    if ref_size == 0:
        accuracy = 0.0
    else:
        accuracy = (C - I + ref_size) / (2.0 * ref_size)
        accuracy = max(0.0, min(1.0, accuracy))

    return accuracy, correct_pitches, missing_pitches, extra_pitches


# Step 0 - make first note start at t = 0.0
# ------------------------------------------------------------------------------
def normalize_start_times(notes):
    """
    Shift all notes so that the first note starts at t=0.
 
    Args:
        notes: list of note dicts, each with at least a "start" key.
 
    Returns:
        A new list of note dicts (copies, not the original objects), with
        every "start" value shifted so notes[0]["start"] == 0. Returns an
        empty list unchanged if notes is empty.
    """
    if not notes:
        return []
 
    # Use the earliest onset rather than assuming notes[0] is the first note.
    first_start = min(note["start"] for note in notes)
 
    shifted_notes = []
    for note in notes:
        # Copy the complete dictionary so that additional metadata is retained.
        note_copy = note.copy()
        # Only modify the copied note's start time.
        note_copy["start"] = note["start"] - first_start
        shifted_notes.append(note_copy)
 
    return shifted_notes

# helper function to build an event dict from a group of notes
def make_event(notes_in_group):
    """
    Build a single event dict from a group of notes.
 
    Args:
        notes_in_group: list of one or more note dicts.
 
    Returns:
        event dict with keys "event_type", "notes", "start". example:
        {
            "event_type": "note" or "chord" depending on the number of notes in the group
            "notes": [                    
                {                         
                    "pitch": int
                    "start": float
                    "duration": float
                },
            ]
            "event_start": float, use the start time of the first note in the group
            "event_duration": float, use the longest duration among all notes in the group
        }
    """
    event_type = "note" if len(notes_in_group) == 1 else "chord"
    return {
        "event_type": event_type,
        "notes": notes_in_group,
        # use the start time of the first note in the group as the event start
        "event_start": notes_in_group[0]["start"], 
        # use the longest duration among all notes in the group as the event duration
        "event_duration": max(note["duration"] for note in notes_in_group), 
    }

# group notes into events based on their start times
def group_notes_into_events(notes, chord_onset_window=DEFAULT_CHORD_ONSET_WINDOW):
    """
    Group a flat list of notes into events. Notes whose start times fall
    within chord_onset_window seconds of each other are placed into the
    same event (i.e. treated as a chord). Notes that are not grouped with
    any other note form a single-note event.

    Args:
        notes: list of dicts, each with keys 'pitch', 'start', 'duration'
        chord_onset_window: float, max time difference (seconds) to be grouped

    Returns:
        events: list of event dicts, each with keys:
            "event_type": "note" if only one note, "chord" if two or more
            "notes": list of note dicts belonging to this event
            "event_start": float, start time of the first note in the group
            "event_duration": float, longest duration among all notes in the group
    """
    if len(notes) == 0:
        return []

    # make sure notes are sorted by start time first
    sorted_notes = sorted(notes, key=lambda n: n["start"])

    events = []
    current_group = [sorted_notes[0]]
    group_start = sorted_notes[0]["start"]

    for note in sorted_notes[1:]:
        # Close enough in time: add to current group (chord)
        if note["start"] - group_start <= chord_onset_window:
            current_group.append(note)
        else:
            # Too far apart: save current chord, start a new group
            event = make_event(current_group)
            events.append(event)
            current_group = [note]
            group_start = note["start"]

    # append the last group
    last_event = make_event(current_group)
    events.append(last_event)

    return events


# Step 1 -- edit-distance alignment to identify missing/extra notes and pitch errors
# ------------------------------------------------------------------------------
def build_cost_matrix(response_events, ref_events, gap_penalty=DEFAULT_GAP_PENALTY):
    """
    Precompute the full (N x M) event-alignment cost matrix using vectorised
    NumPy broadcasting, instead of compute event cost one at a
    time inside a Python loop. This also removes the redundant cost
    computation that used to happen a second time during backtracking.

    Args:
        response_events: list of event dicts (from group_notes_into_events)
        ref_events: list of event dicts (from group_notes_into_events)
        gap_penalty: cost used when event types (note vs chord) do not match

    Returns:
        numpy array of shape (N, M): cost of aligning each response event
        to each reference event.
    """
    N = len(response_events)
    M = len(ref_events)

    # Build simple per-event arrays: is this event a chord, and (if it is a
    # single note) what is its pitch.
    res_is_chord = np.array([event["event_type"] == "chord" for event in response_events])
    ref_is_chord = np.array([event["event_type"] == "chord" for event in ref_events])

    res_pitch = np.array([
        event["notes"][0]["pitch"] if event["event_type"] == "note" else 0
        for event in response_events
    ])
    ref_pitch = np.array([
        event["notes"][0]["pitch"] if event["event_type"] == "note" else 0
        for event in ref_events
    ])

    # Note-vs-note cost: vectorised absolute pitch difference for every pair.
    # Shape (N, 1) - shape (1, M) broadcasts to (N, M)
    note_cost_matrix = np.abs(res_pitch.reshape(N, 1) - ref_pitch.reshape(1, M))

    # Chord-vs-chord cost: build a 12-dim binary pitch-class vector per chord,
    # then compute Hamming distance for every pair using broadcasting.
    res_pitch_vector = np.zeros((N, 12), dtype=int)
    for i in range(N):
        if res_is_chord[i]:
            for note in response_events[i]["notes"]:
                res_pitch_vector[i, note["pitch"] % 12] = 1

    ref_pitch_vector = np.zeros((M, 12), dtype=int)
    for j in range(M):
        if ref_is_chord[j]:
            for note in ref_events[j]["notes"]:
                ref_pitch_vector[j, note["pitch"] % 12] = 1

    # (N, 1, 12) vs (1, M, 12) broadcasts to (N, M, 12); summing the last
    # axis gives the Hamming distance for every (response, ref) pair at once.
    differs = res_pitch_vector.reshape(N, 1, 12) != ref_pitch_vector.reshape(1, M, 12)
    chord_cost_matrix = differs.sum(axis=2)

    # Type-mismatch mask: True where one side is a note and the other a chord.
    type_mismatch = res_is_chord.reshape(N, 1) != ref_is_chord.reshape(1, M)
    both_chords = res_is_chord.reshape(N, 1) & ref_is_chord.reshape(1, M)

    # Combine the three cases -- np.where(condition, true, false)
    cost_matrix = np.where(
        type_mismatch,
        gap_penalty,
        np.where(both_chords, chord_cost_matrix, note_cost_matrix),
    )

    return cost_matrix.astype(int)


def event_alignment_ED(response_events, ref_events, gap_penalty=DEFAULT_GAP_PENALTY):
    """
    Same algorithm as the production event_alignment_ED(), but the N x M
    substitution costs are looked up from a precomputed cost matrix instead
    of being recomputed on every DP cell and again during backtracking.

    Args:
        response_events: list of event dicts from group_notes_into_events
        ref_events: list of event dicts from group_notes_into_events
        gap_penalty: cost of leaving an event unaligned (insertion/deletion)

    Returns:
        operations: list of operation dicts, same format as event_alignment_ED()
        D: accumulated cost matrix, shape (N+1, M+1)
    """
    # if a raw note dict with "pitch"/"start"/"duration" but no "event_type" is
    # passed in, group them into events first. 
    if "event_type" not in response_events[0]:
        response_events = group_notes_into_events(response_events)
    if "event_type" not in ref_events[0]:
        ref_events = group_notes_into_events(ref_events)

    # the rows of D correspond to response events
    N = len(response_events)
    # the columns of D correspond to reference events
    M = len(ref_events)

    # Precompute every substitution cost once
    cost_matrix = build_cost_matrix(response_events, ref_events, gap_penalty)

    # Build the accumulated cost matrix D of size (N+1 x M+1)
    D = np.zeros((N + 1, M + 1), dtype=int)
    
    # Boundary conditions: aligning against an empty sequence means every event
    # is unaligned, so the cost is n (or m) times the gap penalty.
    for n in range(1, N + 1):
        D[n, 0] = n * gap_penalty # n extra response events
    for m in range(1, M + 1):
        D[0, m] = m * gap_penalty # m missing ref events

    # Recursion (accumulated cost / score matrix D):
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            replace_cost = cost_matrix[n - 1, m - 1]
            D[n, m] = min(
                D[n-1, m-1] + replace_cost, # diagonal: match or replacement
                D[n-1, m] + gap_penalty, # vertical: extra event response[n-1]
                D[n, m-1] + gap_penalty, # horizontal: missing response for ref[m-1]
            )

    # Backtrack and classify each transformation op based on movement direction in D
    operations = []
    n, m = N, M
    while n > 0 or m > 0:
        # Boundary conditions: at the top row, only horizontal moves possible
        if n == 0:
            # Missing response for ref[m-1] (deletion)
            operations.append({
                "type": "missing",
                "response_idx": None,
                "reference_idx": m - 1,
                "cost": gap_penalty,
            })
            m -= 1
        # At the leftmost column, only vertical moves possible
        elif m == 0:
            # Extra event response[n-1] (insertion)
            operations.append({
                "type": "extra",
                "response_idx": n - 1,
                "reference_idx": None,
                "cost": gap_penalty,
            })
            n -= 1
        # For all other cases, we can move in any direction (diagonal, vertical, horizontal)
        else:
            replace_cost = cost_matrix[n - 1, m - 1]
            diag = D[n - 1, m - 1] + replace_cost # Diagonal -> match or replacement
            up = D[n - 1, m] + gap_penalty # Vertical -> response[n-1] is extra (insertion)
            left = D[n, m - 1] + gap_penalty # Horizontal -> response is missing for ref[m-1] (deletion)
            min_cost = min(diag, up, left) 

            # classify the transformation ops based on the minimum cost step
            # rule: always prefer diagonal > insertion or deletion
            if min_cost == diag: # Diagonal -> two events are aligned (match or replacement)
                operations.append({
                    "type": "match" if replace_cost == 0 else "replacement",
                    "response_idx": n - 1,
                    "reference_idx": m - 1,
                    "cost": int(replace_cost),
                })
                n, m = n - 1, m - 1
            elif min_cost == up: # Vertical -> response[n-1] is extra (insertion)
                operations.append({
                    "type": "extra",
                    "response_idx": n - 1,
                    "reference_idx": None,
                    "cost": gap_penalty,
                })
                n -= 1
            else: # Horizontal -> response is missing for ref[m-1] (deletion)
                operations.append({
                    "type": "missing",
                    "response_idx": None,
                    "reference_idx": m - 1,
                    "cost": gap_penalty,
                })
                m -= 1

    operations.reverse()
    return operations, D


# Step 2 -- estimate_global_timing and estimate_global_duration_scale
# ------------------------------------------------------------------------------
def estimate_global_timing(operations, response_events, ref_events):
    """
    Estimate the student's overall tempo relative to the reference, by fitting
    a straight line through the matched note start times:
        response_start ≈ scale * ref_start + offset
    where:
        scale > 1 means the student is playing slower overall
        scale < 1 means the student is playing faster overall
        offset captures any constant time shift

    Args:
        operations: list of operation dicts (match/replacement/missing/extra)
        response_events: list of event dicts from response
        ref_events: list of event dicts from reference

    Returns:
        scale: float, estimated tempo ratio (1.0 = same speed as reference)
        offset: float (seconds), estimated constant time shift
    """
    # Collect (ref_start, response_start) pairs from matched/replaced notes only.
    # Missing/extra notes have no pair, so they cannot contribute to the fit.
    ref_starts = []
    response_starts = []
    for op in operations:
        if op["type"] in ("match", "replacement"):
            res = op["response_idx"]
            ref = op["reference_idx"]
            ref_starts.append(ref_events[ref]["event_start"])
            response_starts.append(response_events[res]["event_start"])
 
    # Not enough points for fitting a meaningful line — assume no drift in tempo.
    if len(ref_starts) < 3:
        return 1.0, 0.0
 
    x = np.array(ref_starts, dtype=float)
    y = np.array(response_starts, dtype=float)
 
    # Least-squares line fit: y = scale * x + offset
    scale, offset = np.polyfit(x, y, 1)
 
    return float(scale), float(offset)


def estimate_global_duration_scale(operations, response_events, ref_events):
    """
    Estimate the student's overall note-length scale relative to the reference,
    by fitting a line through the origin:
        response_duration ≈ duration_scale * ref_duration
    where:
        duration_scale > 1 means notes are held longer overall
        duration_scale < 1 means notes are held shorter overall

    Args:
        operations: output of event_alignment_ED()
        response_events: list of student event dicts
        ref_events: list of reference event dicts

    Returns:
        duration_scale (float): estimated duration ratio (1.0 = same as reference)
    """
    ref_durations = []
    response_durations = []
    for op in operations:
        if op["type"] in ("match", "replacement"):
            res = op["response_idx"]
            ref = op["reference_idx"]
            ref_durations.append(ref_events[ref]["event_duration"])
            response_durations.append(response_events[res]["event_duration"])

    if len(ref_durations) < 3:
        return 1.0

    x = np.array(ref_durations, dtype=float)
    y = np.array(response_durations, dtype=float)

    # Least-squares fit through the origin: y = scale * x
    # Closed-form solution: scale = sum(x*y) / sum(x*x)
    duration_scale = float(np.sum(x * y) / np.sum(x * x))

    return duration_scale


# Step 3 -- event_level_feedback
# ------------------------------------------------------------------------------
def event_level_feedback(operations, response_events, ref_events,
                         timing_scale=1.0, timing_offset=0.0,
                         duration_scale=1.0,
                         timing_relative_threshold=TIMING_RELATIVE_THRESHOLD,
                         duration_relative_threshold=DURATION_RELATIVE_THRESHOLD):
    """
    Analyse each aligned event pair (or missing/extra event) and return a
    list of event result dicts.
    For single-note events, pitch evaluation uses absolute pitch difference.
    For chord events, pitch evaluation uses the chord accuracy metric A:
        A = (C - I + |y|) / (2 * |y|)
        where C = correctly played pitch classes, I = extra pitch classes played,
        |y| = number of pitch classes in the reference chord.

    Args:
        operations: list of op dicts (match/replacement/missing/extra)
        response_events: list of event dicts from response
        ref_events: list of event dicts from reference
        timing_scale: float, estimated tempo ratio (1.0 = same speed as reference)
        timing_offset: float (seconds), estimated constant time shift
        duration_scale: float, estimated overall duration ratio
        timing_relative_threshold: float, relative tolerance for timing correctness
        duration_relative_threshold: float, relative tolerance for duration correctness

    Returns:
        event_level_results : list of dicts, each dict contains:

        For note events, each dict has:
            "event_type" -> "note"
            "reference_index" -> int (1-based) or None
            "response_index" -> int (1-based) or None
            "operation_type" -> "match", "replacement", "missing", "extra"
            "pitch_correct" -> bool
            "pitch_diff" -> int (semitones) or None
            "timing_correct" -> bool
            "timing_abs_diff" -> float (seconds) or None
            "timing_relative_diff" -> float or None
            "duration_correct" -> bool
            "duration_abs_diff" -> float (seconds) or None
            "duration_relative_diff" -> float or None
 
        For chord events, each dict has:
            "event_type" -> "chord"
            "reference_index" -> int (1-based) or None
            "response_index" -> int (1-based) or None
            "operation_type" -> "match", "replacement", "missing", "extra"
            "chord_name_ref" -> str e.g. "C major", or None
            "chord_name_res" -> str e.g. "C minor", or None
            "chord_accuracy" -> float (0 to 1) or None
            "correct_pitches" -> list of pitch class ints or None
            "missing_pitches" -> list of pitch class ints or None
            "extra_pitches" -> list of pitch class ints or None
            "timing_correct" -> bool
            "timing_abs_diff" -> float (seconds) or None
            "timing_relative_diff" -> float or None
            "duration_correct" -> bool
            "duration_abs_diff" -> float (seconds) or None
            "duration_relative_diff" -> float or None
    """
    # Compute IOI for each reference note: ioi[m] = ref_notes[m]["start"] - ref_notes[m-1]["start"]
    # floor at 0.05s to avoid division by zero issues
    ref_ioi = [None] * len(ref_events)
    for m in range(1, len(ref_events)):
        interval = ref_events[m]["event_start"] - ref_events[m - 1]["event_start"]
        ref_ioi[m] = max(interval, 0.05)
 
    event_level_results = []

    for op in operations:
        res_idx = op["response_idx"]
        ref_idx = op["reference_idx"]
        op_type = op["type"]

        # Determine event type from whichever side is available
        if ref_idx is not None:
            event_type = ref_events[ref_idx]["event_type"]
        else:
            event_type = response_events[res_idx]["event_type"]

        # Missing/extra notes: no pitch/timing/duration comparison is possible,
        # so all the numeric fields are set to None.
        if op_type in ("missing", "extra"):
            if event_type == "note":
                event_level_results.append({
                    "event_type": "note",
                    "reference_index": (ref_idx + 1) if ref_idx is not None else None,
                    "response_index":  (res_idx + 1) if res_idx is not None else None,
                    "operation_type": op_type,
                    "pitch_correct": False,
                    "pitch_diff": None,
                    "timing_correct": False,
                    "timing_abs_diff": None,
                    "timing_relative_diff": None,
                    "duration_correct": False,
                    "duration_abs_diff": None,
                    "duration_relative_diff": None,
                })
            else:
                if ref_idx is not None:
                    chord_name = identify_chord_name(ref_events[ref_idx]["notes"])
                else:
                    chord_name = identify_chord_name(
                        response_events[res_idx]["notes"]
                    )
                event_level_results.append({
                    "event_type": "chord",
                    "reference_index": (ref_idx + 1) if ref_idx is not None else None,
                    "response_index":  (res_idx + 1) if res_idx is not None else None,
                    "operation_type": op_type,
                    "chord_name_ref": chord_name if op_type == "missing" else None,
                    "chord_name_res": chord_name if op_type == "extra" else None,
                    "chord_accuracy": None,
                    "correct_pitches": None,
                    "missing_pitches": None,
                    "extra_pitches": None,
                    "timing_correct": False,
                    "timing_abs_diff": None,
                    "timing_relative_diff": None,
                    "duration_correct": False,
                    "duration_abs_diff": None,
                    "duration_relative_diff": None,
                })
        else:
            # Aligned event pair (match or replacement)
            res_event = response_events[res_idx]
            ref_event = ref_events[ref_idx]

            # Timing — residual after removing the global tempo trend
            predicted_start = timing_scale * ref_event["event_start"] + timing_offset
            timing_abs_diff = abs(res_event["event_start"] - predicted_start)
            if ref_idx == 0:
                # First note will start at 0, so no difference.
                timing_relative_diff = None
                timing_correct = True
            else:
                ioi = ref_ioi[ref_idx]
                timing_relative_diff = timing_abs_diff / ioi
                timing_correct = (timing_relative_diff <= timing_relative_threshold)

            # Duration — residual after removing the global duration-scale trend
            predicted_duration = duration_scale * ref_event["event_duration"]
            duration_abs_diff = abs(res_event["event_duration"] - predicted_duration)
            ref_dur = max(ref_event["event_duration"], 0.05) # floor at 0.05s to avoid division by zero issues
            duration_relative_diff = duration_abs_diff / ref_dur
            duration_correct = (duration_relative_diff <= duration_relative_threshold)
            if event_type == "note":
                # For single-note events, pitch correctness is based on absolute pitch difference.
                pitch1 = res_event["notes"][0]["pitch"]
                pitch2 = ref_event["notes"][0]["pitch"]
                pitch_diff = int(abs(pitch1 - pitch2))
                event_level_results.append({
                    "event_type": "note",
                    "reference_index": ref_idx + 1,
                    "response_index":  res_idx + 1,
                    "operation_type": op_type,
                    "pitch_correct": (pitch_diff == 0),
                    "pitch_diff": pitch_diff,
                    "timing_correct": timing_correct,
                    "timing_abs_diff": timing_abs_diff,
                    "timing_relative_diff": timing_relative_diff,
                    "duration_correct": duration_correct,
                    "duration_abs_diff": duration_abs_diff,
                    "duration_relative_diff": duration_relative_diff,
                })
            else:
                # For chord events, pitch correctness is based on the chord accuracy metric.
                accuracy, correct_pcs, missing_pcs, extra_pcs = (
                    compute_chord_accuracy(
                        ref_event["notes"], res_event["notes"]
                    )
                )
                event_level_results.append({
                    "event_type": "chord",
                    "reference_index": ref_idx + 1,
                    "response_index":  res_idx + 1,
                    "operation_type": op_type,
                    "chord_name_ref": identify_chord_name(ref_event["notes"]),
                    "chord_name_res": identify_chord_name(res_event["notes"]),
                    "chord_accuracy": accuracy,
                    "correct_pitches": correct_pcs,
                    "missing_pitches": missing_pcs,
                    "extra_pitches": extra_pcs,
                    "timing_correct": timing_correct,
                    "timing_abs_diff": timing_abs_diff,
                    "timing_relative_diff": timing_relative_diff,
                    "duration_correct": duration_correct,
                    "duration_abs_diff": duration_abs_diff,
                    "duration_relative_diff": duration_relative_diff,
                })

    return event_level_results


# Step 4 -- compute_stats
# ------------------------------------------------------------------------------
def compute_stats(event_level_results, ref_events, timing_scale=1.0,
                  timing_offset=0.0, duration_scale=1.0):
    """
    Compute summary counts and correctness booleans from event-level feedback.

    Args:
        event_level_results: list of dicts, output of event_level_feedback()
        ref_events: list of reference event dicts
        timing_scale: float, from estimate_global_timing()
        timing_offset: float, from estimate_global_timing()
        duration_scale: float, from estimate_global_duration_scale()

    Returns:
        stats: dict with keys:
            "pitch_all_aligned_correct" -> bool, True if all note pitches are
                                            correct AND all chord accuracies are 1.0
            "timing_all_correct" -> bool
            "duration_all_correct" -> bool
            "total_notes_in_reference" -> int
            "total_notes_missing" -> int
            "total_notes_extra" -> int
            "total_notes_wrong_pitch" -> int
            "total_notes_wrong_timing" -> int
            "total_notes_wrong_duration" -> int
            "total_notes_correct" -> int
            "timing_scale" -> float
            "timing_offset" -> float
            "duration_scale" -> float
            "total_chords_in_reference" -> int
            "total_chords_missing" -> int
            "total_chords_extra" -> int
            "total_chords_correct" -> int (accuracy == 1.0)
            "total_chords_imperfect" -> int (0.0 < accuracy < 1.0)
            "total_chords_wrong" -> int (accuracy == 0.0)
    """
    note_events  = [n for n in event_level_results if n["event_type"] == "note"]
    chord_events = [ch for ch in event_level_results if ch["event_type"] == "chord"]
    
    ref_note_count  = sum(1 for n in ref_events if n["event_type"] == "note")
    ref_chord_count = sum(1 for ch in ref_events if ch["event_type"] == "chord")
    
    paired_notes  = [
        n for n in note_events
        if n["operation_type"] in ("match", "replacement")
    ]
    paired_chords = [
        ch for ch in chord_events
        if ch["operation_type"] in ("match", "replacement")
    ]
    all_paired = [
        e for e in event_level_results
        if e["operation_type"] in ("match", "replacement")
    ]

    stats = {
        "pitch_all_aligned_correct": (
            all(n["pitch_correct"] for n in paired_notes)
            and all(
                ch["chord_accuracy"] == 1.0
                for ch in paired_chords
                if ch["chord_accuracy"] is not None
            )
        ),
        "timing_all_correct": all(n["timing_correct"] for n in all_paired),
        "duration_all_correct": all(n["duration_correct"] for n in all_paired),
        "total_notes_in_reference": ref_note_count,
        "total_notes_missing": sum(1 for n in note_events if n["operation_type"] == "missing"),
        "total_notes_extra": sum(1 for n in note_events if n["operation_type"] == "extra"),
        "total_notes_wrong_pitch": sum(1 for n in paired_notes if not n["pitch_correct"]),
        "total_notes_wrong_timing": sum(1 for n in paired_notes if not n["timing_correct"]),
        "total_notes_wrong_duration": sum(1 for n in paired_notes if not n["duration_correct"]),
        "total_notes_correct": sum(1 for n in paired_notes
            if n["pitch_correct"] and n["timing_correct"] and n["duration_correct"]
        ),
        "timing_scale": timing_scale,
        "timing_offset": timing_offset,
        "duration_scale": duration_scale,
        "total_chords_in_reference": ref_chord_count,
        "total_chords_missing": sum(1 for ch in chord_events if ch["operation_type"] == "missing"),
        "total_chords_extra": sum(1 for ch in chord_events if ch["operation_type"] == "extra"),
        "total_chords_correct": sum(
            1 for ch in paired_chords
            if ch["chord_accuracy"] is not None and ch["chord_accuracy"] == 1.0
        ),
        "total_chords_imperfect": sum(
            1 for ch in paired_chords
            if ch["chord_accuracy"] is not None
            and ch["chord_accuracy"] > 0.0
            and ch["chord_accuracy"] < 1.0
        ),
        "total_chords_wrong": sum(
            1 for ch in paired_chords
            if ch["chord_accuracy"] is not None and ch["chord_accuracy"] == 0.0
        ),
    }
    return stats


# Step 5 -- generate_feedback_message
# ------------------------------------------------------------------------------
def generate_feedback_message(event_details, response_events, ref_events, stats,
                               global_slow_threshold=GLOBAL_SLOW_THRESHOLD,
                               global_fast_threshold=GLOBAL_FAST_THRESHOLD):
    """
    Generate human-readable feedback messages for the student.

    Part 1 - Overview: summary of timing trend, duration trend, and total counts
             of each error type (pitch / missing / extra).
    Part 2 - Note Detail: pitch, timing, duration errors per note
    Part 3 - Chord Detail: errors per chord

    Args:
        event_details: list of dicts, output of event_level_feedback()
        response_events: list of event dicts from group_notes_into_events
        ref_events: list of event dicts from group_notes_into_events
        stats: dict, output of compute_stats()
        global_slow_threshold: timing_scale above this triggers "too slow" message
        global_fast_threshold: timing_scale below this triggers "too fast" message

    Returns:
        feedback_message (str)
    """
    note_events  = [n for n in event_details if n["event_type"] == "note"]
    chord_events = [ch for ch in event_details if ch["event_type"] == "chord"]

    paired_notes  = [
        n for n in note_events
        if n["operation_type"] in ("match", "replacement")
    ]
    paired_chords = [
        ch for ch in chord_events
        if ch["operation_type"] in ("match", "replacement")
    ]

    timing_scale = stats["timing_scale"]
    timing_offset = stats["timing_offset"]
    duration_scale = stats["duration_scale"]

    overview_messages = []
    note_detail_messages = []
    chord_detail_messages = []

    # ---------- Part 1: Overview ----------
    # Tempo: acceptable / too slow / too fast 
    timing_pct = abs(timing_scale - 1.0) * 100
    duration_pct = abs(duration_scale - 1.0) * 100
    if timing_scale > 1:
        timing_direction = "behind"
    elif timing_scale < 1:
        timing_direction = "ahead of"
    else:
        timing_direction = "the same as"

    if duration_scale > 1:
        duration_direction = "longer than"
    elif duration_scale < 1:
        duration_direction = "shorter than"
    else:
        duration_direction = "the same as"

    if timing_scale > global_slow_threshold:
        overview_messages.append(
            f"Overall, your tempo is slower than the reference "
            f"(timing is about {timing_pct:.0f}% {timing_direction} the reference in general while "
            f"notes are held about {duration_pct:.0f}% {duration_direction} the reference). "
            f"No worries! You will get better when you practice more to get more familiar with it!"
        )
    elif timing_scale < global_fast_threshold:
        overview_messages.append(
            f"Overall, your tempo is faster than the reference "
            f"(timing is about {timing_pct:.0f}% {timing_direction} the reference in general while "
            f"notes are held about {duration_pct:.0f}% {duration_direction} the reference). "
            f"Don't rush even if you are confident in your performance." 
            f"Slow down and give each note its full value."
        )
    else:
        overview_messages.append(
            f"Timing: your overall tempo is within an acceptable range. Good job! "
            f"The timing is about {timing_pct:.0f}% {timing_direction} the reference in general while "
            f"notes are held about {duration_pct:.0f}% {duration_direction} than the reference."
        )

    # Wrong notes pitch counts
    if stats["total_notes_wrong_pitch"] > 0:
        s = "is" if stats["total_notes_wrong_pitch"] == 1 else "are"
        note_word = "note" if stats["total_notes_wrong_pitch"] == 1 else "notes"
        overview_messages.append(
            f"There {s} {stats['total_notes_wrong_pitch']} {note_word} played with the wrong pitch."
        )
    else:
        overview_messages.append("There are no pitch errors. Well done!")
    # Missing notes counts
    if stats["total_notes_missing"] > 0:
        s = "is" if stats["total_notes_missing"] == 1 else "are"
        note_word = "note" if stats["total_notes_missing"] == 1 else "notes"
        overview_messages.append(
            f"There {s} {stats['total_notes_missing']} {note_word} you missed from the reference."
        )
    else:
        overview_messages.append("There are no missing notes. Great!")
    # Extra notes counts
    if stats["total_notes_extra"] > 0:
        s = "is" if stats["total_notes_extra"] == 1 else "are"
        note_word = "note" if stats["total_notes_extra"] == 1 else "notes"
        overview_messages.append(
            f"There {s} {stats['total_notes_extra']} extra {note_word} played during practice. "
            f"You may need to adjust your fingering or hand position to avoid extra notes."
        )
    else:
        overview_messages.append("There are no extra notes. Good job!")
    # Chord errors counts
    if stats["total_chords_in_reference"] > 0:
        total = stats["total_chords_in_reference"]
        correct = stats["total_chords_correct"]
        imperfect = stats["total_chords_imperfect"]
        wrong = stats["total_chords_wrong"]
        overview_messages.append(
            f"Chords: {correct}/{total} correct, "
            f"{imperfect}/{total} imperfect (some notes missing or extra), "
            f"{wrong}/{total} completely wrong."
        )
        if stats["total_chords_missing"] > 0:
            c_word = "chord" if stats["total_chords_missing"] == 1 else "chords"
            overview_messages.append(
                f"{stats['total_chords_missing']} {c_word} missed."
            )
        if stats["total_chords_extra"] > 0:
            c_word = "chord" if stats["total_chords_extra"] == 1 else "chords"
            overview_messages.append(
                f"{stats['total_chords_extra']} extra {c_word} played."
            )

    # ---------- Part 2: Note Detail ----------
    # Missing / extra notes
    for n in note_events:
        if n["operation_type"] == "missing":
            ref_zero_based = n["reference_index"] - 1
            pitch = ref_events[ref_zero_based]["notes"][0]["pitch"]
            note_detail_messages.append(
                f"Note {n['reference_index']} (pitch {pitch}) is missing in your performance."
            )
        elif n["operation_type"] == "extra":
            res_zero_based = n["response_index"] - 1
            extra = response_events[res_zero_based]["notes"][0]["pitch"]
            note_detail_messages.append(
                f"Extra note played: pitch {extra} "
                f"at t={response_events[res_zero_based]['event_start']:.2f}s ")
    # Pitch errors
    for n in paired_notes:
        if not n["pitch_correct"]:
            ref_zero_based = n["reference_index"] - 1
            res_zero_based = n["response_index"] - 1
            ref_p = ref_events[ref_zero_based]["notes"][0]["pitch"]
            res_p = response_events[res_zero_based]["notes"][0]["pitch"]
            note_detail_messages.append(
                f"Note {n['reference_index']}: wrong pitch — "
                f"expected {ref_p}, played {res_p} "
                f"({n['pitch_diff']} semitone(s) off)."
            )
    # Local timing errors - these are residuals after removing the global timing trend
    for n in paired_notes:
        if not n["timing_correct"]:
            note_detail_messages.append(
                f"Note {n['reference_index']}: timing is off by "
                f"{n['timing_abs_diff']:.2f}s "
                f"({n['timing_relative_diff'] * 100:.0f}% of the expected note interval), "
                f"after accounting for the overall tempo trend."
            )
    # Local duration errors — these are residuals after removing the global duration trend
    for n in paired_notes:
        if not n["duration_correct"]:
            direction = "longer" if n["duration_abs_diff"] > 0 else "shorter"
            duration_pct_err = abs(n["duration_relative_diff"]) * 100
            note_detail_messages.append(
                f"Note {n['reference_index']}: duration is "
                f"{abs(n['duration_abs_diff']):.2f}s {direction} than the reference "
                f"({duration_pct_err:.0f}% off) after accounting for the overall duration trend."
            )

    # ---------- Part 3: Chord Detail ----------
    # Missing / extra chords
    for ch in chord_events:
        if ch["operation_type"] == "missing":
            chord_detail_messages.append(
                f"Chord {ch['reference_index']} ({ch['chord_name_ref']}) "
                f"is missing in your performance."
            )
        elif ch["operation_type"] == "extra":
            chord_detail_messages.append(
                f"Extra chord played: {ch['chord_name_res']} "
                f"at event position {ch['response_index']}."
            )
    # Chord accuracy errors
    for ch in paired_chords:
        if ch["chord_accuracy"] is not None and ch["chord_accuracy"] < 1.0:
            accuracy_pct = round(ch["chord_accuracy"] * 100)
            message = (
                f"Chord {ch['reference_index']} "
                f"(expected {ch['chord_name_ref']}, you played {ch['chord_name_res']}): "
                f"{accuracy_pct}% accurate. "
            )
            if ch["missing_pitches"]:
                missing_names = [PITCH_CLASS_NAMES[pitch % 12] for pitch in ch["missing_pitches"]]
                message = message + "Missing note(s): " + ", ".join(missing_names) + ". "
            if ch["extra_pitches"]:
                extra_names = [PITCH_CLASS_NAMES[pitch % 12] for pitch in ch["extra_pitches"]]
                message = message + "Extra note(s) played: " + ", ".join(extra_names) + "."
            chord_detail_messages.append(message)
    # Local timing errors for chords
    for ch in paired_chords:
        if not ch["timing_correct"] and ch["timing_relative_diff"] is not None:
            chord_detail_messages.append(
                f"Chord {ch['reference_index']}: timing is off by "
                f"{ch['timing_abs_diff']:.2f}s "
                f"({ch['timing_relative_diff'] * 100:.0f}% of the expected interval)."
            )

    all_messages = ["Overview: "] + overview_messages

    if note_detail_messages:
        all_messages = all_messages + ["", "Note Detail:"] + note_detail_messages
    else:
        all_messages = all_messages + ["", "All melody notes played correctly!!"]
    
    if stats["total_chords_in_reference"] > 0:
        if chord_detail_messages:
            all_messages = (
                all_messages + ["", "Chord Detail:"] + chord_detail_messages
            )
        else:
            all_messages = all_messages + ["", "Great performance! No further issues on chords found."]
        
    return "\n".join(all_messages)


def polished_feedback_message(event_details, response_events, ref_events, stats,
                               global_slow_threshold=GLOBAL_SLOW_THRESHOLD,
                               global_fast_threshold=GLOBAL_FAST_THRESHOLD):
    """
    Generate concise, practice-oriented feedback that aims to:
        1. summarise current performance level qualitatively
        2. provide encouraging actionable advice for improvement
        3. suggest a measurable goal for the next attempt with a main focus area
    Individual note and chord errors remain available in event_details but
    will not list each of them in the message.

    Args:
        event_details: list of dicts, output of event_level_feedback()
        response_events: list of event dicts from group_notes_into_events
        ref_events: list of event dicts from group_notes_into_events
        stats: dict, output of compute_stats()
        global_slow_threshold: timing_scale above this triggers "too slow" message
        global_fast_threshold: timing_scale below this triggers "too fast" message

    Returns:
        feedback_message (str)
    """
    note_events  = [n for n in event_details if n["event_type"] == "note"]
    chord_events = [ch for ch in event_details if ch["event_type"] == "chord"]

    paired_notes  = [
        n for n in note_events
        if n["operation_type"] in ("match", "replacement")
    ]
    paired_chords = [
        ch for ch in chord_events
        if ch["operation_type"] in ("match", "replacement")
    ]

    total_reference_notes = stats["total_notes_in_reference"]
    total_reference_chords = stats["total_chords_in_reference"]
    total_reference_events = total_reference_notes + total_reference_chords

    # --------------- summarise current performance level ---------------
    # compute note-level accuracy metrics for pitch, timing, and duration
    # -------------------------------------------------------------------
    if total_reference_notes > 0:
        note_pitch_correct = (total_reference_notes 
                              - stats["total_notes_wrong_pitch"] - stats["total_notes_missing"])
        note_pitch_accuracy = note_pitch_correct / total_reference_notes

        note_timing_correct = (total_reference_notes 
                          - stats["total_notes_wrong_timing"] - stats["total_notes_missing"])
        note_timing_accuracy = note_timing_correct / total_reference_notes
     
        note_duration_correct = (total_reference_notes
                            - stats["total_notes_wrong_duration"] - stats["total_notes_missing"])
        note_duration_accuracy = note_duration_correct / total_reference_notes 
    else:
        note_pitch_accuracy = None
        note_timing_accuracy = None
        note_duration_accuracy = None
 
    paired_chord_accuracies = [event["chord_accuracy"] 
                               for event in paired_chords 
                               if event["chord_accuracy"] is not None]
    # Missing chords were never matched, treat as 0 accuracy for the purpose of computing median chord accuracy
    paired_chord_accuracies = paired_chord_accuracies + [0.0] * stats["total_chords_missing"]
    if paired_chord_accuracies:
        median_chord_accuracy = float(np.median(paired_chord_accuracies))
    else:
        median_chord_accuracy = None

    # summary qualitative feedback messages based on the accuracy metrics
    # -------------------------------------------------------------------
    current_performance_messages = []

    if note_pitch_accuracy is not None:
        if note_pitch_accuracy >= 0.90:
            current_performance_messages.append(
                "Great! Most notes were played correctly, " \
                "you've got a good grasp of the melody.")
        elif note_pitch_accuracy >= 0.70:
            current_performance_messages.append(
                "Many notes were correct, although a few passages " \
                "still need more careful practice. Try slowing down in " \
                "these sections and checking each note before gradually " \
                "returning to the intended tempo."
            )
        else:
            current_performance_messages.append(
                "Note accuracy needs more focused practice. " 
                "Practice one short passage at slower tempo, " 
                "check each note carefully, and then speed up to the correct tempo. " \
                "Move on to the next passage when you feel confident with current one."
            )

    if note_timing_accuracy is not None:
        if note_timing_accuracy >= 0.90:
            current_performance_messages.append(
                "Great timing consistency between notes! "
            )
        elif note_timing_accuracy >= 0.70:
            current_performance_messages.append(
                "The spacing between notes was mostly consistent, although " \
                "a few passages were less steady. Practicing these sections " \
                "with a slower, regular beat may help make the rhythm more even."
            )
        else:
            current_performance_messages.append(
                "Timing consistency needs more practice, listen carefully " \
                "for notes that arrive too early or too late. " \
                "You may find it helpful to practice with a metronome. "
            )
        # do not comment on duration for now, as it may be due to transcription errors

    if total_reference_chords > 0:
        if median_chord_accuracy is None or median_chord_accuracy < 0.70:
            current_performance_messages.append(
                            "Simultaneous notes are often hard to play correctly at the beginning. " \
                            "Pay attention to the fingering and hand position when playing these chords. " \
                            "Practice those sections slowly and carefully, and check that all required notes sound together. " \
                            "You may find it helpful to practice each chord separately before reconnecting the surrounding passage."
                        )
        elif median_chord_accuracy >= 0.90:
            current_performance_messages.append(
                "Nice! The chord notes were accurate overall."
            )
        elif median_chord_accuracy >= 0.70:
            current_performance_messages.append(
                "Many chord notes were correct, although some chords included " \
                "missing or additional notes. Practice each difficult chord " \
                "separately and check that all required notes sound together."
            )

    # overall tempo feedback based on timing and duration scale factors
    # -------------------------------------------------------------------
    timing_scale = stats["timing_scale"]
    duration_scale = stats["duration_scale"]
    if timing_scale > global_slow_threshold:
        tempo_message = (
            "Your overall tempo was slower than the reference. It is a good idea to " \
            "play slowly while learning, so focus on maintaining a steady tempo " \
            "before gradually increasing the speed."
        )
    elif timing_scale < global_fast_threshold:
        tempo_message = (
            "Your overall tempo was faster than the reference. Try slowing " \
            "down slightly so that each note can be played clearly and the rhythm remains controlled."
        )
    else:
        tempo_message = (
            "Well done! Your overall tempo was close to the reference. Keep up the good work!"
        )

    # missing and extra
    # ------------------------------------------------------------------
    total_missing_events = (stats["total_notes_missing"] + stats["total_chords_missing"])
    total_extra_events = (stats["total_notes_extra"] + stats["total_chords_extra"])
    total_completeness_errors = (total_missing_events + total_extra_events)
    if total_reference_events > 0:
        completeness_error_rate = total_completeness_errors / total_reference_events
    else:
        completeness_error_rate = 0.0

    if total_completeness_errors == 0:
        completeness_messages = (
            "You completed the performance without missing or adding any " \
            "notes or chords. Well done!"
        )
    elif completeness_error_rate <= 0.10:
        completeness_messages = (
            "The performance was mostly complete, with only a few missing " \
            "or additional notes or chords. Review the affected passages " \
            "slowly and check your fingering before playing them again."
        )
    else:
        completeness_messages = (
            "No worries! It is common to miss or play extra notes when learning a new piece, " \
            "especially difficult passages. You can slow down in your next practice and pay " \
            "more attention to your fingering and hand position. ")

    # -------------- suggest a measurable goal and a focus area --------------
    scores = {}
    if note_pitch_accuracy is not None:
        scores["pitch"] = note_pitch_accuracy
    if note_timing_accuracy is not None:
        scores["timing"] = note_timing_accuracy
    # if note_duration_accuracy is not None:
        # scores["duration"] = note_duration_accuracy
    if median_chord_accuracy is not None:
        scores["chords"] = median_chord_accuracy

    if scores:
        main_focus = min(scores, key=scores.get)
    else:
        main_focus = None

    if main_focus == "pitch":
        focus_message = (
            "For your next attempt, choose one short challenging passage and practice it slowly. " \
            "Aim to play this phrase correctly three times in a row before increasing the tempo."
        )
    elif main_focus == "timing":
        focus_message = (
            "Practice with a slower, steady beat, preferably using a metronome. " \
            "Aim to keep the spacing between the notes evenly three times in a row. " \
            "Then gradually increase the tempo to the correct speed."
        )
    elif main_focus == "chords":
        focus_message = (
            "For your next attempt, choose one difficult chord and aim to make " \
            "all required notes sound together correctly three times in a row. " \
            "Then gradually connect it to the surrounding passage."
        )
    else:
        focus_message = ("For your next attempt, choose one short challenging section and " \
            "aim to play it confidently three times in a row.")

    all_messages = [
        "Practice Summary",
        "",
        " ".join(current_performance_messages),
        "",
        "Tempo",
        tempo_message,
        "",
        "Performance Completeness", 
        completeness_messages,
        "",
        "Main Practice Focus",
        focus_message,
        "",
        "Keep up the good work and enjoy your music journey!",
    ]
        
    return "\n".join(all_messages)


# FeedbackResult class
# ------------------------------------------------------------------------------
class FeedbackResult:
    """
    Container for all outputs of compare_performance_ED().
    Using a class (instead of returning a tuple) makes unit tests much clearer:
        result = compare_performance_ED(response, reference)
        assert result.is_correct == False
        assert result.stats["total_notes_missing"] == 1
        assert "missing" in result.feedback_message

    Attributes
    ----------
    is_correct : bool
        True only if every note/chord has correct pitch,
        timing and duration, with no missing or extra events.
    stats : dict
        Aggregate counts. See compute_stats() for all keys.
    event_details : list of dicts
        Per-event analysis, one dict per alignment operation.
        Each dict has an "event_type" key ("note" or "chord"),
        plus type-specific fields. See event_level_feedback() for details.
    feedback_message : str
        Human-readable feedback string, ready to display to the student.
    operations : list of dicts
        Raw alignment operations from event_alignment_ED().
        Kept here so visualisation helpers (plot_cost_matrix etc.) can use them.
    D : numpy.ndarray
        Accumulated cost matrix from the alignment step.
    """

    def __init__(self, is_correct, stats, event_details,
                 feedback_message, operations, D):
        self.is_correct       = is_correct
        self.stats            = stats
        self.event_details    = event_details
        self.feedback_message = feedback_message
        self.operations       = operations
        self.D                = D
 
    def __repr__(self):
        return (
            "FeedbackResult(is_correct=" + str(self.is_correct) + ", "
            "stats=" + str(self.stats) + ")"
        )


# Pipeline
# ------------------------------------------------------------------------------
def compare_performance_ED(responseMIDI, refMIDI,
                            gap_penalty=DEFAULT_GAP_PENALTY,
                            timing_relative_threshold=TIMING_RELATIVE_THRESHOLD,
                            duration_relative_threshold=DURATION_RELATIVE_THRESHOLD,
                            global_slow_threshold=GLOBAL_SLOW_THRESHOLD,
                            global_fast_threshold=GLOBAL_FAST_THRESHOLD,
                            chord_onset_window=DEFAULT_CHORD_ONSET_WINDOW):
    """
    Full pipeline: normalisation -> grouping -> alignment -> global trends
                   -> event-level evaluation -> summary statistics -> feedback.
 
    Args:
        responseMIDI: student MIDI dict with key "notes"
        refMIDI: reference MIDI dict with key "notes"
        gap_penalty: cost of an unaligned event
        timing_relative_threshold: see event_level_feedback()
        duration_relative_threshold: see event_level_feedback()
        global_slow_threshold: see generate_feedback_message()
        global_fast_threshold: see generate_feedback_message()
        chord_onset_window: float (seconds), notes within this window are
            grouped into a chord. Default 0.050 (50ms). Teacher-configurable.
 
    Returns:
        FeedbackResult object containing all analysis results
    """
    # Step 0: Normalise start times
    response_notes = normalize_start_times(responseMIDI["notes"])
    ref_notes = normalize_start_times(refMIDI["notes"])
    # Group notes into events (single notes or chords)
    response_events = group_notes_into_events(response_notes, chord_onset_window)
    ref_events = group_notes_into_events(ref_notes, chord_onset_window)

    # Step 1: Align events using edit distance
    operations, D = event_alignment_ED(response_events, ref_events, gap_penalty)

    # Step 2: Estimate the overall tempo trend (global timing and duration trends)
    timing_scale, timing_offset = estimate_global_timing(
        operations, response_events, ref_events
    )
    duration_scale = estimate_global_duration_scale(
        operations, response_events, ref_events
    )

    # Step 3: Event details feedback
    event_details = event_level_feedback(
        operations, response_events, ref_events,
        timing_scale=timing_scale,
        timing_offset=timing_offset,
        duration_scale=duration_scale,
        timing_relative_threshold=timing_relative_threshold,
        duration_relative_threshold=duration_relative_threshold,
    )

    # Step 4: Compute summary statistics
    stats = compute_stats(
        event_details, ref_events,
        timing_scale=timing_scale,
        timing_offset=timing_offset,
        duration_scale=duration_scale,
    )

    # Step 5: Generate human-readable feedback
    feedback_message = polished_feedback_message(
        event_details, response_events, ref_events, stats,
        global_slow_threshold=global_slow_threshold,
        global_fast_threshold=global_fast_threshold,
    )

    # Step 6: Overall pass/fail judgement
    is_correct = (
        stats["total_notes_missing"] == 0
        and stats["total_notes_extra"] == 0
        and stats["total_chords_missing"] == 0
        and stats["total_chords_extra"] == 0
        and stats["pitch_all_aligned_correct"]
        and stats["timing_all_correct"]
        and stats["duration_all_correct"]
    )

    return FeedbackResult(
        is_correct=is_correct,
        stats=stats,
        event_details=event_details,
        feedback_message=feedback_message,
        operations=operations,
        D=D,
    )