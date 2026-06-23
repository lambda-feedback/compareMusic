"""
compare_MIDI.py
================
Core MIDI evaluation pipeline for the compareMusic evaluation function.

Pipeline overview (called in order by compare_performance_ED):
    Step 0 -- normalize_start_times     (make first note start at t = 0.0)
    Step 1 -- note_alignment_ED         (edit-distance alignment)
    Step 2 -- estimate_global_timing    (linear regression for tempo drift)
              estimate_global_duration_scale
    Step 3 -- note_level_feedback       (per-note pitch / timing / duration check)
    Step 4 -- compute_stats             (summary counts)
    Step 5 -- generate_feedback_message (human-readable text)
"""


import numpy as np

# Default thresholds / parameters
# Teachers can override any of these via the params dict in evaluation_function.
# ------------------------------------------------------------------------------
# Gap penalty: cost of leaving a note unaligned (insertion/deletion)
DEFAULT_GAP_PENALTY = 6

# Timing: |response_start - predicted_start| / IOI must be below this.
# e.g. 0.20 means the start can be off by up to 20% of the inter-onset interval.
TIMING_RELATIVE_THRESHOLD = 0.20

# Duration: |response_dur / ref_dur - 1| must be below this.
# e.g. 0.25 means the student's duration can be off by up to 25% of the reference.
DURATION_RELATIVE_THRESHOLD = 0.25

# Thresholds that trigger a global tempo comment in the overview.
GLOBAL_SLOW_THRESHOLD = 1.15   # timing_scale > 1.15  -> "overall too slow"
GLOBAL_FAST_THRESHOLD = 0.85   # timing_scale < 0.85  -> "overall too fast"


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
 
    first_start = notes[0]["start"]
 
    shifted_notes = []
    for note in notes:
        # Create a copy of the note dict with the "start" time shifted
        note_copy = {
            "pitch": note["pitch"],
            "start": note["start"] - first_start,
            "duration": note["duration"],
        }
        shifted_notes.append(note_copy)
 
    return shifted_notes


# Step 1 -- edit-distance alignment to identify missing/extra notes and pitch errors
# ------------------------------------------------------------------------------
def compute_cost(note1, note2):
    """
    Cost of aligning (replacing) one note with another, based on pitch.
 
    cost = 0: pitches are identical (a 'match'). 
    cost > 0: different pitches (a 'replacement')
 
    Args:
        note1: dict with keys "pitch" (int), "start" (float), "duration" (float)
        note2: dict with keys "pitch" (int), "start" (float), "duration" (float)
 
    Returns:
        int: cost value >= 0 (lower means more similar pitch)
    """
    return int(abs(note1["pitch"] - note2["pitch"]))


def note_alignment_ED(response_notes, ref_notes, gap_penalty=DEFAULT_GAP_PENALTY):
    """
    Align notes using edit distance (ED). 
    The ED allows for insertions and deletions, which can be useful for 
    evaluating musical practice containing missing/extra notes.
    
    Args:
        response_notes: The student's response MIDI notes to evaluate
        ref_notes: The reference MIDI note
        gap_penalty: cost of leaving a note unaligned (insertion/deletion)
 
    Returns:
        operations: list of transformation ops dicts, in order from first note to last:
            {'type': 'match' or 'replacement' or 'missing' or 'extra', 
            'response_idx': int or None, 
            'reference_idx': int or None, 
            'cost': int}
        D: accumulated cost matrix, shape (N+1, M+1)
    """
    # the rows of D correspond to response notes
    N = len(response_notes)
    # the columns of D correspond to reference notes
    M = len(ref_notes)

    # Build the accumulated cost matrix D of size (N+1 x M+1)
    D = np.zeros((N + 1, M + 1), dtype=int)
    
    # Boundary conditions: aligning against an empty sequence means every note
    # is unaligned, so the cost is n (or m) times the gap penalty.
    for n in range(1, N + 1):
        D[n, 0] = n * gap_penalty # n extra response notes
    for m in range(1, M + 1):
        D[0, m] = m * gap_penalty # m missing ref notes

    # Recursion (accumulated cost / score matrix D):
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            replace_cost = compute_cost(response_notes[n-1], ref_notes[m-1])
            D[n, m] = min(
                D[n-1, m-1] + replace_cost, # diagonal: match or replacement
                D[n-1, m] + gap_penalty, # vertical: extra note response[n-1]
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
            # Extra note response[n-1] (insertion)
            operations.append({
                "type": "extra",
                "response_idx": n - 1,
                "reference_idx": None,
                "cost": gap_penalty,
            })
            n -= 1
        # For all other cases, we can move in any direction (diagonal, vertical, horizontal)
        else:
            replace_cost = compute_cost(response_notes[n - 1], ref_notes[m - 1])
            diag = D[n - 1, m - 1] + replace_cost # diagonal: match or replacement
            up   = D[n - 1, m] + gap_penalty # vertical: extra note response[n-1]
            left = D[n, m - 1] + gap_penalty # horizontal: missing response for ref[m-1]
            min_cost = min(diag, up, left) # find the minimum cost step
            # classify the transformation ops based on the minimum cost step
            if min_cost == diag: # Diagonal -> two notes are aligned (match or replacement)
                operations.append({
                    "type": "match" if replace_cost == 0 else "replacement",
                    "response_idx": n - 1,
                    "reference_idx": m - 1,
                    "cost": replace_cost,
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

    operations.reverse()  # Reverse to get ops in order from first note to last
    return operations, D


# Step 2 -- estimate_global_timing and estimate_global_duration_scale
# ------------------------------------------------------------------------------
def estimate_global_timing(operations, response_notes, ref_notes):
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
        response_notes: list of note dicts from response
        ref_notes: list of note dicts from reference

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
            ref_starts.append(ref_notes[op["reference_idx"]]["start"])
            response_starts.append(response_notes[op["response_idx"]]["start"])
 
    # Not enough points for fitting a meaningful line — assume no drift in tempo.
    if len(ref_starts) < 3:
        return 1.0, 0.0
 
    x = np.array(ref_starts, dtype=float)
    y = np.array(response_starts, dtype=float)
 
    # Least-squares line fit: y = scale * x + offset
    scale, offset = np.polyfit(x, y, 1)
 
    return float(scale), float(offset)

def estimate_global_duration_scale(operations, response_notes, ref_notes):
    """
    Estimate the student's overall note-length scale relative to the reference,
    by fitting a line through the origin:
        response_duration ≈ duration_scale * ref_duration
    where:
        duration_scale > 1 means notes are held longer overall
        duration_scale < 1 means notes are held shorter overall

    Args:
        operations: output of note_alignment_ED()
        response_notes: list of student note dicts
        ref_notes: list of reference note dicts

    Returns:
        duration_scale (float): estimated duration ratio (1.0 = same as reference)
    """
    ref_durations = []
    response_durations = []
    for op in operations:
        if op["type"] in ("match", "replacement"):
            ref_durations.append(ref_notes[op["reference_idx"]]["duration"])
            response_durations.append(response_notes[op["response_idx"]]["duration"])

    if len(ref_durations) < 3:
        return 1.0

    x = np.array(ref_durations, dtype=float)
    y = np.array(response_durations, dtype=float)

    # Least-squares fit through the origin: y = scale * x
    # Closed-form solution: scale = sum(x*y) / sum(x*x)
    duration_scale = float(np.sum(x * y) / np.sum(x * x))

    return duration_scale


# Step 3 -- note_level_feedback
# ------------------------------------------------------------------------------
def note_level_feedback(operations, response_notes, ref_notes, 
                        timing_scale=1.0, timing_offset=0.0, duration_scale=1.0,
                        timing_relative_threshold=TIMING_RELATIVE_THRESHOLD,
                        duration_relative_threshold=DURATION_RELATIVE_THRESHOLD):
    """
    Analyse each aligned note pair (or missing/extra event) and return a list
    of note result dicts.

    Args:
        operations: list of op dicts (match/replacement/missing/extra)
        response_notes: list of note dicts from response
        ref_notes: list of note dicts from reference
        timing_scale: float, estimated tempo ratio (1.0 = same speed as reference)
        timing_offset: float (seconds), estimated constant time shift
        duration_scale: float, estimated overall duration ratio
        timing_relative_threshold: float, relative tolerance for timing correctness
        duration_relative_threshold: float, relative tolerance for duration correctness

    Returns:
        note_level_results : list of dicts, each dict contains:
            "reference_index" -> int (1-based) or None if operation_type = extra
            "response_index" -> int (1-based) or None if operation_type = missing
            "operation_type" -> str: "match", "replacement", "missing", or "extra"
            "pitch_correct" -> bool
            "pitch_diff" -> int (semitones) or None if operation_type = missing/extra
            "timing_correct" -> bool 
            "timing_abs_diff" -> float (seconds) or None if operation_type = missing/extra
            “timing_relative_diff” -> float (seconds) or None if operation_type = missing/extra
            "duration_correct" -> bool
            "duration_abs_diff"-> float (seconds) or None if operation_type = missing/extra
            "duration_relative_diff" -> float (seconds) or None if operation_type = missing/extra
    """
    # Compute IOI for each reference note: ioi[m] = ref_notes[m]["start"] - ref_notes[m-1]["start"]
    # floor at 0.05s to avoid division by zero issues
    ref_ioi = [None] * len(ref_notes)
    for m in range(1, len(ref_notes)):
        interval = ref_notes[m]["start"] - ref_notes[m - 1]["start"]
        ref_ioi[m] = max(interval, 0.05)
 
    note_level_results = []

    for op in operations:
        res_idx = op["response_idx"]
        ref_idx = op["reference_idx"]
        op_type = op["type"]

        # Missing/extra notes: no pitch/timing/duration comparison is possible,
        # so all the numeric fields are set to None.
        if op_type in ("missing", "extra"):
            note_level_results.append({
                "reference_index": (ref_idx + 1) if ref_idx is not None else None,
                "response_index": (res_idx + 1) if res_idx is not None else None,
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
            # Matched (aligned) note pair
            res_note = response_notes[res_idx]
            ref_note = ref_notes[ref_idx]

            ## Pitch
            pitch_diff = int(abs(res_note["pitch"] - ref_note["pitch"]))
            pitch_correct = (pitch_diff == 0)

            # Timing — residual after removing the global tempo trend
            predicted_start = timing_scale * ref_note["start"] + timing_offset
            timing_abs_diff = abs(res_note["start"] - predicted_start)
            if ref_idx == 0:
                # First note will start at 0, so no difference.
                timing_relative_diff = None
                timing_correct = True
            else:
                ioi = ref_ioi[ref_idx]
                timing_relative_diff = timing_abs_diff / ioi
                timing_correct = (timing_relative_diff <= timing_relative_threshold)

            # Duration — residual after removing the global duration-scale trend
            predicted_duration = duration_scale * ref_note["duration"]
            duration_abs_diff = res_note["duration"] - predicted_duration
            ref_dur = max(ref_note["duration"], 0.05) # floor at 0.05s to avoid division by zero issues
            duration_relative_diff = duration_abs_diff / ref_dur
            duration_correct = (abs(duration_relative_diff) <= duration_relative_threshold)

            note_level_results.append({
                "reference_index": ref_idx + 1,
                "response_index": res_idx + 1,
                "operation_type": op_type,
                "pitch_correct": pitch_correct,
                "pitch_diff": pitch_diff,
                "timing_correct": timing_correct,
                "timing_abs_diff": timing_abs_diff,
                "timing_relative_diff": timing_relative_diff,
                "duration_correct": duration_correct,
                "duration_abs_diff": duration_abs_diff,
                "duration_relative_diff": duration_relative_diff,
            })

    return note_level_results


# Step 4 -- compute_stats
# ------------------------------------------------------------------------------
def compute_stats(note_details, ref_notes, timing_scale=1.0,
                  timing_offset=0.0, duration_scale=1.0):
    """
    Compute summary counts and correctness booleans from note-level feedback.

    Args:
        note_details: list of dicts, output of note_level_feedback()
        ref_notes: list of reference note dicts
        timing_scale: float, from estimate_global_timing()
        timing_offset: float, from estimate_global_timing()
        duration_scale: float, from estimate_global_duration_scale()

    Returns:
        stats: dict with keys:
            "pitch_all_correct" -> bool
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
    """
    paired = [n for n in note_details
              if n["operation_type"] in ("match", "replacement")]

    stats = {
        "pitch_all_correct": all(n["pitch_correct"] for n in paired),
        "timing_all_correct": all(n["timing_correct"] for n in paired),
        "duration_all_correct": all(n["duration_correct"] for n in paired),
        "total_notes_in_reference": len(ref_notes),
        "total_notes_missing": sum(1 for n in note_details if n["operation_type"] == "missing"),
        "total_notes_extra": sum(1 for n in note_details if n["operation_type"] == "extra"),
        "total_notes_wrong_pitch": sum(1 for n in paired if not n["pitch_correct"]),
        "total_notes_wrong_timing": sum(1 for n in paired if not n["timing_correct"]),
        "total_notes_wrong_duration": sum(1 for n in paired if not n["duration_correct"]),
        "total_notes_correct": sum(1 for n in paired
            if n["pitch_correct"] and n["timing_correct"] and n["duration_correct"]
        ),
        "timing_scale": timing_scale,
        "timing_offset": timing_offset,
        "duration_scale": duration_scale,
    }

    return stats


# Step 5 -- generate_feedback_message
# ------------------------------------------------------------------------------
def generate_feedback_message(note_details, response_notes, ref_notes, stats,
                               global_slow_threshold=GLOBAL_SLOW_THRESHOLD,
                               global_fast_threshold=GLOBAL_FAST_THRESHOLD):
    """
    Generate human-readable feedback messages for the student.

    Part 1 - Overview: summary of timing trend, duration trend, and total counts
             of each error type (pitch / missing / extra).
    Part 2 - Detail: indicate exactly which notes have which problems.

    Args:
        note_details: list of dicts, output of note_level_feedback()
        response_notes: list of student note dicts
        ref_notes: list of reference note dicts
        stats: dict, output of compute_stats()
        global_slow_threshold: timing_scale above this triggers "too slow" message
        global_fast_threshold: timing_scale below this triggers "too fast" message

    Returns:
        feedback_message (str)
    """
    paired = []
    for n in note_details:
        if n["operation_type"] in ("match", "replacement"):
            paired.append(n)

    timing_scale   = stats["timing_scale"]
    timing_offset  = stats["timing_offset"]
    duration_scale = stats["duration_scale"]

    overview_messages = []
    detail_messages = []

    # ---------- Part 1: Overview ----------
    # Tempo: acceptable / too slow / too fast ---
    timing_pct = abs(timing_scale - 1.0) * 100
    duration_pct = abs(duration_scale - 1.0) * 100
    timing_direction = "behind" if timing_scale > 1.0 else "ahead of"
    duration_direction = "longer" if duration_scale > 1.0 else "shorter"

    if timing_scale > global_slow_threshold:
        overview_messages.append(
            f"Overall, your tempo is slower than the reference "
            f"(timing is about {timing_pct:.0f}% {timing_direction} the reference in general while "
            f"notes are held about {duration_pct:.0f}% {duration_direction} than the reference). "
            f"No worries! You will get better when you practice more to get more familiar with it!"
        )
    elif timing_scale < global_fast_threshold:
        overview_messages.append(
            f"Overall, your tempo is faster than the reference "
            f"(timing is about {timing_pct:.0f}% {timing_direction} the reference in general while "
            f"notes are held about {duration_pct:.0f}% {duration_direction} than the reference). "
            f"Don't rush even if you are confident in your performance." 
            f"Slow down and give each note its full value."
        )
    else:
        overview_messages.append(
            f"Timing: your overall tempo is within an acceptable range. Good job! "
            f"The timing is about {timing_pct:.0f}% {timing_direction} the reference in general while "
            f"notes are held about {duration_pct:.0f}% {duration_direction} than the reference."
        )

    # Wrong pitch counts
    if stats["total_notes_wrong_pitch"] > 0:
        s = "is" if stats["total_notes_wrong_pitch"] == 1 else "are"
        note_word = "note" if stats["total_notes_wrong_pitch"] == 1 else "notes"
        overview_messages.append(
            f"There {s} {stats['total_notes_wrong_pitch']} {note_word} played with the wrong pitch."
        )
    else:
        overview_messages.append("There are no pitch errors. Well done!")
    # Missing counts
    if stats["total_notes_missing"] > 0:
        s = "is" if stats["total_notes_missing"] == 1 else "are"
        note_word = "note" if stats["total_notes_missing"] == 1 else "notes"
        overview_messages.append(
            f"There {s} {stats['total_notes_missing']} {note_word} you missed from the reference."
        )
    else:
        overview_messages.append("There are no missing notes. Great!")
    # Extra counts
    if stats["total_notes_extra"] > 0:
        s = "is" if stats["total_notes_extra"] == 1 else "are"
        note_word = "note" if stats["total_notes_extra"] == 1 else "notes"
        overview_messages.append(
            f"There {s} {stats['total_notes_extra']} extra {note_word} played during practice. "
            f"You may need to adjust your fingering or hand position to avoid extra notes."
        )
    else:
        overview_messages.append("There are no extra notes. Good job!")

    # ---------- Part 2: Detail ----------
    # Missing / extra notes
    for n in note_details:
        if n["operation_type"] == "missing":
            ref_zero_based = n["reference_index"] - 1
            pitch = ref_notes[ref_zero_based]["pitch"]
            detail_messages.append(
                f"Note {n['reference_index']} (pitch {pitch}) is missing in your performance."
            )
        elif n["operation_type"] == "extra":
            res_zero_based = n["response_index"] - 1
            extra = response_notes[res_zero_based]
            detail_messages.append(
                f"Extra note played: pitch {extra['pitch']} at t={extra['start']:.2f}s ")

    # Pitch errors
    for n in paired:
        if not n["pitch_correct"]:
            ref_zero_based = n["reference_index"] - 1
            res_zero_based = n["response_index"] - 1
            ref_p = ref_notes[ref_zero_based]["pitch"]
            res_p = response_notes[res_zero_based]["pitch"]
            detail_messages.append(
                f"Note {n['reference_index']}: wrong pitch — "
                f"expected {ref_p}, played {res_p} "
                f"({n['pitch_diff']} semitone(s) off)."
            )

    # Local timing errors - these are residuals after removing the global timing trend
    for n in paired:
        if not n["timing_correct"]:
            detail_messages.append(
                f"Note {n['reference_index']}: timing is off by {n['timing_abs_diff']:.2f}s "
                f"({n['timing_relative_diff'] * 100:.0f}% of the expected note interval), "
                f"after accounting for the overall tempo trend."
            )

    # Local duration errors — these are residuals after removing the global duration trend
    for n in paired:
        if not n["duration_correct"]:
            direction = "longer" if n["duration_abs_diff"] > 0 else "shorter"
            ref_zero_based = n["reference_index"] - 1
            ref_dur = ref_notes[ref_zero_based]["duration"]
            duration_pct = abs(n["duration_relative_diff"]) * 100
            detail_messages.append(
                f"Note {n['reference_index']}: duration is {abs(n['duration_abs_diff']):.2f}s "
                f"{direction} than the reference (i.e. "
                f"{duration_pct:.0f}% off) after accounting for the overall duration trend "
            )

    all_messages = ["Overview: "] + overview_messages

    if detail_messages:
        all_messages = all_messages + ["", "Detail: "] + detail_messages
    else:
        all_messages = all_messages + ["", "Great performance! No further issues found."]
        
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
        True only if every note is perfectly matched on pitch, timing, and duration.
    stats : dict
        Aggregate counts — see compute_stats() for the full key list.
    note_details : list of dicts
        Per-note analysis, one dict per alignment operation.
        Each dict has the keys described in note_level_feedback().
    feedback_message : str
        Human-readable feedback string, ready to display to the student.
        see generate_feedback_message() for details.
    operations : list of dicts
        Raw alignment operations from note_alignment_ED().
        Kept here so visualisation helpers (plot_cost_matrix etc.) can use them.
    D : numpy.ndarray
        Accumulated cost matrix from the alignment step.
    """

    def __init__(self, is_correct, stats, note_details,
                 feedback_message, operations, D):
        self.is_correct       = is_correct
        self.stats            = stats
        self.note_details     = note_details
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
                            global_fast_threshold=GLOBAL_FAST_THRESHOLD):
    """
    Full pipeline: normalisation -> alignment -> estimate global trends
                   -> note-level evaluation -> summary statistics -> feedback.

    Args:
        responseMIDI: student MIDI dict with key "notes"
        refMIDI: reference MIDI dict with key "notes"
        gap_penalty: cost of an unaligned note
        timing_relative_threshold: see note_level_feedback()
        duration_relative_threshold: see note_level_feedback()
        global_slow_threshold: see generate_feedback_message()
        global_fast_threshold: see generate_feedback_message()

    Returns:
        FeedbackResult object containing all analysis results
    """
    # Step 0: Normalise start times
    response_notes = normalize_start_times(responseMIDI["notes"])
    ref_notes      = normalize_start_times(refMIDI["notes"])

    # Step 1: Align notes using edit distance
    operations, D = note_alignment_ED(response_notes, ref_notes, gap_penalty)

    # Step 2: Estimate the overall tempo trend
    timing_scale, timing_offset = estimate_global_timing(
        operations, response_notes, ref_notes
    )
    duration_scale = estimate_global_duration_scale(
        operations, response_notes, ref_notes
    )

    # Step 3: Note-level evaluation
    note_details = note_level_feedback(
        operations, response_notes, ref_notes,
        timing_scale=timing_scale,
        timing_offset=timing_offset,
        duration_scale=duration_scale,
        timing_relative_threshold=timing_relative_threshold,
        duration_relative_threshold=duration_relative_threshold,
    )

    # Step 4: Compute summary statistics
    stats = compute_stats(
        note_details, ref_notes,
        timing_scale=timing_scale,
        timing_offset=timing_offset,
        duration_scale=duration_scale,
    )

    # Step 5: Generate the human-readable feedback text
    feedback_message = generate_feedback_message(
        note_details, response_notes, ref_notes, stats,
        global_slow_threshold=global_slow_threshold,
        global_fast_threshold=global_fast_threshold,
    )

    # Step 6: Overall pass/fail judgement
    is_correct = (
        stats["total_notes_missing"] == 0
        and stats["total_notes_extra"] == 0
        and stats["pitch_all_correct"]
        and stats["timing_all_correct"]
        and stats["duration_all_correct"]
    )

    return FeedbackResult(
        is_correct=is_correct,
        stats=stats,
        note_details=note_details,
        feedback_message=feedback_message,
        operations=operations,
        D=D,
    )