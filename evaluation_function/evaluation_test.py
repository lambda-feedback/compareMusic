"""
evaluation_tests.py
===================
Unit tests for the compareMusic evaluation function.
    Read the docs on how to use unittest here:
    https://docs.python.org/3/library/unittest.html
Run locally with:  python -m pytest evaluation_test.py -v
 
Sections
--------
0. Helper: make_midi
1. Tests for helper functions get_pitch_class_set, identify_chord_name, compute_chord_accuracy
2. Tests for normalize_start_times
3. Tests for group_notes_into_events
4. Tests for compute_note_cost and compute_event_cost
5. Tests for event_alignment_ED  (covers note-only, chord-only cases and mixed cases)
6. Tests for estimate_global_timing and estimate_global_duration_scale
7. Tests for event_level_feedback and compute_stats  (covers note-only and chord-only cases)
8. Tests for evaluation_function (Lambda Feedback integration)
9. Tests for parameter overrides
"""


import unittest
from .compare_MIDI import (
    normalize_start_times,
    group_notes_into_events,
    make_event,
    compute_note_cost,
    compute_event_cost,
    get_pitch_class_set,
    identify_chord_name,
    compute_chord_accuracy,
    event_alignment_ED,
    estimate_global_timing,
    estimate_global_duration_scale,
    event_level_feedback,
    compute_stats,
    compare_performance_ED,
    DEFAULT_GAP_PENALTY,
    TIMING_RELATIVE_THRESHOLD,
    DURATION_RELATIVE_THRESHOLD,
    GLOBAL_SLOW_THRESHOLD,
    GLOBAL_FAST_THRESHOLD,
    DEFAULT_CHORD_ONSET_WINDOW,
)
from .evaluation import evaluation_function

# 0. Helper: create a minimal MIDI dictionary for testing
# ------------------------------------------------------------------------------
def make_midi(pitches, starts, durations):
    notes = []
    for i in range(len(pitches)):
        notes.append({
            "pitch": pitches[i],
            "start": starts[i],
            "duration": durations[i],
        })
    return {"notes": notes}


# 1. Tests for helper functions get_pitch_class_set, identify_chord_name, compute_chord_accuracy
# ------------------------------------------------------------------------------
class TestChordHelpers(unittest.TestCase):
 
    def test_get_pitch_class_set_single_octave(self):
        notes = [{"pitch": 60}, {"pitch": 64}, {"pitch": 67}]
        result = get_pitch_class_set(notes)
        assert result == {0, 4, 7}
 
    def test_get_pitch_class_set_cross_octave(self):
        # C4=60 and C5=72 both map to pitch class 0
        notes = [{"pitch": 60}, {"pitch": 72}]
        result = get_pitch_class_set(notes)
        assert result == {0}
 
    def test_identify_c_major(self):
        notes = [
            {"pitch": 60},  # C
            {"pitch": 64},  # E
            {"pitch": 67},  # G
        ]
        assert identify_chord_name(notes) == "C major"
 
    def test_identify_unknown_chord(self):
        # A random cluster that does not match any template
        notes = [{"pitch": 60}, {"pitch": 61}, {"pitch": 62}]
        assert identify_chord_name(notes) == "unknown chord"
 
    def test_perfect_accuracy_is_one(self):
        ref = [{"pitch": 60}, {"pitch": 64}, {"pitch": 67}]
        res = [{"pitch": 60}, {"pitch": 64}, {"pitch": 67}]
        accuracy, correct, missing, extra = compute_chord_accuracy(ref, res)
        assert accuracy == 1.0
        assert missing == []
        assert extra == []
 
    def test_completely_wrong_notes(self):
        # Response has no overlap with reference
        ref = [{"pitch": 60}, {"pitch": 64}, {"pitch": 67}]
        res = [{"pitch": 61}, {"pitch": 65}, {"pitch": 69}]
        accuracy, correct, missing, extra = compute_chord_accuracy(ref, res)
        assert accuracy == 0.0
        assert len(correct) == 0
 
    def test_partial_match_returns_score_between_0_and_1(self):
        # C major ref (0,4,7) vs C-minor response (0,3,7) -- one note off
        ref = [{"pitch": 60}, {"pitch": 64}, {"pitch": 67}]
        res = [{"pitch": 60}, {"pitch": 63}, {"pitch": 67}]
        accuracy, correct, missing, extra = compute_chord_accuracy(ref, res)
        assert 0.0 < accuracy < 1.0
        assert len(correct) == 2
        assert 4 in missing   # pitch class 4 (E) is missing
        assert 3 in extra     # pitch class 3 (D#) is extra


# 2. Tests for normalize_start_times
# ------------------------------------------------------------------------------
class TestNormalizeStartTimes(unittest.TestCase):

    def test_first_note_starts_at_zero(self):
        notes = make_midi([60, 62], [1.0, 1.5], [0.5, 0.5])["notes"]
        result = normalize_start_times(notes)
        assert result[0]["start"] == 0.0
 
    def test_relative_gaps_preserved(self):
        notes = make_midi([60, 62], [1.0, 1.6], [0.5, 0.5])["notes"]
        result = normalize_start_times(notes)
        assert abs(result[1]["start"] - 0.6) < 0.0001
 
    def test_pitch_and_duration_unchanged(self):
        notes = make_midi([64], [2.0], [0.8])["notes"]
        result = normalize_start_times(notes)
        assert result[0]["pitch"] == 64
        assert result[0]["duration"] == 0.8

    def test_already_starts_at_zero_unchanged(self):
        notes = make_midi([60, 62], [0.0, 0.5], [0.4, 0.4])["notes"]
        result = normalize_start_times(notes)
        assert result[0]["start"] == 0.0
        assert abs(result[1]["start"] - 0.5) < 0.0001


# 3. Tests for group_notes_into_events
# ------------------------------------------------------------------------------
class TestGroupNotesIntoEvents(unittest.TestCase):
 
    def test_single_notes_not_grouped(self):
        # Notes 0.5s apart -- should form separate events
        notes = make_midi([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4])["notes"]
        events = group_notes_into_events(notes)
        assert len(events) == 3
        for event in events:
            assert len(event["notes"]) == 1
 
    def test_three_note_chord_grouped_correctly(self):
        # C major: all three notes at the same start time
        notes = make_midi([60, 64, 67], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])["notes"]
        events = group_notes_into_events(notes)
        assert len(events) == 1
        assert events[0]["event_type"] == "chord"
        assert len(events[0]["notes"]) == 3
 
    def test_notes_within_window_grouped(self):
        # Notes 30ms apart -- within the default 50ms window -> one chord event
        notes = make_midi([60, 64], [0.00, 0.03], [0.5, 0.5])["notes"]
        events = group_notes_into_events(notes, chord_onset_window=0.05)
        assert len(events) == 1
        assert events[0]["event_type"] == "chord"
        assert len(events[0]["notes"]) == 2
 
    def test_notes_outside_window_not_grouped(self):
        # Notes 100ms apart -- outside the default 50ms window
        notes = make_midi([60, 64], [0.00, 0.10], [0.5, 0.5])["notes"]
        events = group_notes_into_events(notes, chord_onset_window=0.05)
        assert len(events) == 2
        for event in events:
            assert event["event_type"] == "note"
            assert len(event["notes"]) == 1
 
    def test_event_start_equals_first_note_start(self):
        notes = make_midi([60, 64], [0.01, 0.02], [0.5, 0.5])["notes"]
        events = group_notes_into_events(notes, chord_onset_window=0.05)
        assert abs(events[0]["event_start"] - 0.010) < 0.0001
 
    def test_event_duration_equals_max_note_duration(self):
        notes = make_midi([60, 64], [0.0, 0.0], [0.4, 0.6])["notes"]
        events = group_notes_into_events(notes)
        assert abs(events[0]["event_duration"] - 0.6) < 0.0001 
 
    def test_custom_window_zero_means_no_grouping(self):
        notes = make_midi([60, 64], [0.0, 0.001], [0.5, 0.5])["notes"]
        events = group_notes_into_events(notes, chord_onset_window=0.0)
        assert len(events) == 2


# 4. Tests for compute_note_cost and compute_event_cost
# ------------------------------------------------------------------------------
class TestCostComputations(unittest.TestCase):

    def test_note_vs_note_different_pitch(self):
        e1 = make_event(make_midi([60], [0.0], [0.5])["notes"])
        e2 = make_event(make_midi([65], [0.0], [0.5])["notes"])
        assert compute_event_cost(e1, e2) == 5

    def test_chord_vs_chord_one_note_different(self):
        # C major (0,4,7) vs C-minor (0,3,7): one pitch differs -> Hamming = 2
        e1 = make_event(make_midi([60, 64, 67], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])["notes"])
        e2 = make_event(make_midi([60, 63, 67], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])["notes"])
        cost = compute_event_cost(e1, e2)
        assert cost == 2
 
    def test_type_mismatch_returns_gap_penalty(self):
        # note vs chord -- should return the gap_penalty regardless of pitches
        note_event  = make_event(make_midi([60], [0.0], [0.5])["notes"])
        chord_event = make_event(make_midi([60, 64], [0.0, 0.0],  [0.5, 0.5])["notes"])
        cost = compute_event_cost(note_event, chord_event, gap_penalty=10)
        assert cost == 10


# 5. Tests for event_alignment_ED
# ------------------------------------------------------------------------------
class TestEventAlignmentED(unittest.TestCase):
 
    def test_perfect_match_all_match_ops(self):
        ref = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        res = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        ref_events = group_notes_into_events(ref["notes"])
        res_events = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_events, ref_events)
        types = [op["type"] for op in operations]
        assert all(t == "match" for t in types)
 
    def test_missing_note_detected(self):
        # pitch 62 missing in response
        ref = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        res = make_midi([60, 64], [0, 1.0], [0.4, 0.4])
        ref_events = group_notes_into_events(ref["notes"])
        res_events = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_events, ref_events)
        types = [op["type"] for op in operations]
        assert "missing" in types
 
    def test_extra_note_detected(self):
        # extra pitch 62 in response
        ref = make_midi([60, 64], [0, 1.0], [0.4, 0.4])
        res = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        ref_events = group_notes_into_events(ref["notes"])
        res_events = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_events, ref_events)
        types = [op["type"] for op in operations]
        assert "extra" in types
 
    def test_wrong_pitch_is_replacement(self):
        ref = make_midi([60, 62], [0, 0.5], [0.4, 0.4])
        res = make_midi([60, 65], [0, 0.5], [0.4, 0.4])
        ref_events = group_notes_into_events(ref["notes"])
        res_events = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_events, ref_events)
        replacements = [op for op in operations if op["type"] == "replacement"]
        assert len(replacements) == 1

    def test_identical_chords_are_matched(self):
        # Two identical C major chords
        ref = make_midi([60, 64, 67, 60, 64, 67], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.5] * 6)
        res = make_midi([60, 64, 67, 60, 64, 67], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.5] * 6)
        ref_events = group_notes_into_events(ref["notes"])
        res_events = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_events, ref_events)
        types = [op["type"] for op in operations]
        assert all(t == "match" for t in types)

    def test_different_chords_are_replacements(self):
        # C major ref, C-minor response (E replaced by Eb)
        ref = make_midi([60, 64, 67], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        res = make_midi([60, 63, 67], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        ref_events = group_notes_into_events(ref["notes"])
        res_events = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_events, ref_events)
        assert operations[0]["type"] == "replacement"

    def test_mix_note_and_chord_aligned_correctly(self):
        # Reference: one single note (C4) then one C major chord.
        # Response: matches exactly.
        # Checks that note and chord events are both aligned as "match" in the same sequence.
        ref = make_midi([60, 60, 64, 67], [0.0, 1.0, 1.0, 1.0], [0.4, 0.5, 0.5, 0.5])
        res = make_midi([60, 60, 64, 67], [0.0, 1.0, 1.0, 1.0], [0.4, 0.5, 0.5, 0.5])
        ref_events = group_notes_into_events(ref["notes"])
        res_events = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_events, ref_events)
        types = [op["type"] for op in operations]
        assert all(t == "match" for t in types)
        # First event is a note, second is a chord
        assert ref_events[0]["event_type"] == "note"
        assert ref_events[1]["event_type"] == "chord"
 
    def test_ops_are_in_forward_order(self):
        ref = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        res = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        ref_events = group_notes_into_events(ref["notes"])
        res_events = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_events, ref_events)
        ref_indices = [op["reference_idx"] for op in operations if op["reference_idx"] is not None]
        assert ref_indices == sorted(ref_indices)
 
    def test_cost_matrix_shape(self):
        ref = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        res = make_midi([60, 62], [0, 0.5], [0.4, 0.4])
        ref_events = group_notes_into_events(ref["notes"])
        res_events = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_events, ref_events)
        assert D.shape == (len(res_events) + 1, len(ref_events) + 1)


# 6. Tests for estimate_global_timing and estimate_global_duration_scale
# ------------------------------------------------------------------------------
class TestTrendEstimations(unittest.TestCase):
 
    def test_perfect_timing_scale_is_one(self):
        ref = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        res = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        ref_ev = group_notes_into_events(ref["notes"])
        res_ev = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_ev, ref_ev)
        scale, offset = estimate_global_timing(operations, res_ev, ref_ev)
        assert abs(scale - 1.0) < 0.01

    def test_faster_playing_scale_less_than_one(self):
        ref = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        res = make_midi([60, 62, 64, 65], [0, 0.4, 0.8, 1.2], [0.4] * 4)
        ref_ev = group_notes_into_events(ref["notes"])
        res_ev = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_ev, ref_ev)
        scale, offset = estimate_global_timing(operations, res_ev, ref_ev)
        assert scale < 1.0

    def test_slower_playing_with_mixed_note_and_chord(self):
        # One note, one C major chord, then one note.
        # The response is played consistently 20% slower.
        # Global timing scale should therefore be greater than 1.0.
        ref = make_midi(
            [60, 60, 64, 67, 72],
            [0.0, 1.0, 1.0, 1.0, 2.0],
            [0.4, 0.5, 0.5, 0.5, 0.4]
        )
        res = make_midi(
            [60, 60, 64, 67, 72],
            [0.0, 1.2, 1.2, 1.2, 2.4],
            [0.4, 0.5, 0.5, 0.5, 0.4]
        )
        ref_ev = group_notes_into_events(ref["notes"])
        res_ev = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_ev, ref_ev)
        scale, offset = estimate_global_timing(operations, res_ev, ref_ev)
        assert scale > 1.0
 
    def test_perfect_duration_scale_is_one(self):
        ref = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        res = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        ref_ev = group_notes_into_events(ref["notes"])
        res_ev = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_ev, ref_ev)
        dur_scale = estimate_global_duration_scale(operations, res_ev, ref_ev)
        assert abs(dur_scale - 1.0) < 0.01

    def test_longer_durations_scale_greater_than_one(self):
        ref = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        res = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.6] * 4)
        ref_ev = group_notes_into_events(ref["notes"])
        res_ev = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_ev, ref_ev)
        dur_scale = estimate_global_duration_scale(operations, res_ev, ref_ev)
        assert dur_scale > 1.0

    def test_fewer_than_3_matched_returns_defaults(self):
        # Only 2 notes -- should return (1.0, 0.0)
        ref = make_midi([60, 62], [0, 0.5], [0.4, 0.4])
        res = make_midi([60, 62], [0, 0.5], [0.4, 0.4])
        ref_events = group_notes_into_events(ref["notes"])
        res_events = group_notes_into_events(res["notes"])
        operations, D = event_alignment_ED(res_events, ref_events)
        scale, offset = estimate_global_timing(operations, res_events, ref_events)
        dur_scale = estimate_global_duration_scale(operations, res_events, ref_events)
        assert scale == 1.0
        assert offset == 0.0
        assert dur_scale == 1.0


# 7. Tests for event_level_feedback and compute_stats
# ------------------------------------------------------------------------------
class TestComparePerformanceED(unittest.TestCase):
 
    def test_consistent_tempo_not_flagged_per_event(self):
        # A student playing consistently 20% slower should NOT get
        # event-level timing warnings -- only a global tempo comment.
        ref = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        res = make_midi([60, 62, 64, 65], [0, 0.6, 1.2, 1.8], [0.4] * 4)
        result = compare_performance_ED(res, ref)
        for n in result.event_details:
            if n["operation_type"] in ("match", "replacement"):
                assert n["timing_correct"] == True
 
    def test_single_late_note_flagged(self):
        ref = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        res = make_midi([60, 62, 64, 65], [0, 0.5, 1.8, 1.5], [0.4] * 4)
        result = compare_performance_ED(res, ref)
        flagged = [n for n in result.event_details if not n["timing_correct"]]
        assert len(flagged) > 0
 
    def test_pitch_error_recorded_correctly(self):
        ref = make_midi([60, 62], [0, 0.5], [0.4, 0.4])
        res = make_midi([60, 65], [0, 0.5], [0.4, 0.4])
        result = compare_performance_ED(res, ref)
        replacements = [n for n in result.event_details if n["operation_type"] == "replacement"]
        assert len(replacements) == 1
        assert replacements[0]["pitch_diff"] == 3
        assert replacements[0]["pitch_correct"] == False
 
    def test_all_correct_stats(self):
        midi = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        result = compare_performance_ED(midi, midi)
        assert result.stats["total_notes_missing"] == 0
        assert result.stats["total_notes_extra"] == 0
        assert result.stats["total_notes_wrong_pitch"] == 0
        assert result.stats["pitch_all_aligned_correct"] == True
 
    def test_missing_note_counted(self):
        ref = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        res = make_midi([60, 64],     [0, 1.0],      [0.4, 0.4])
        result = compare_performance_ED(res, ref)
        assert result.stats["total_notes_missing"] == 1
 
    def test_extra_note_counted(self):
        ref = make_midi([60, 64],     [0, 1.0],      [0.4, 0.4])
        res = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        result = compare_performance_ED(res, ref)
        assert result.stats["total_notes_extra"] == 1
 
    def test_total_notes_in_reference(self):
        ref = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        res = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        result = compare_performance_ED(res, ref)
        assert result.stats["total_notes_in_reference"] == 3

    def test_identical_chords_is_correct(self):
        # Two chords: C major at t=0, F-major at t=1
        midi = make_midi([60, 64, 67, 65, 69, 72],
                         [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                         [0.5] * 6)
        result = compare_performance_ED(midi, midi)
        assert result.is_correct == True
        assert result.stats["total_chords_in_reference"] == 2
        assert result.stats["total_chords_correct"] == 2
 
    def test_missing_chord_counted(self):
        # Reference has C major and F-major; response only has C major
        ref = make_midi([60, 64, 67, 65, 69, 72],
                        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.5] * 6)
        res = make_midi([60, 64, 67], [0.0, 0.0, 0.0], [0.5] * 3)
        result = compare_performance_ED(res, ref)
        assert result.stats["total_chords_missing"] == 1
 
    def test_extra_chord_counted(self):
        ref = make_midi([60, 64, 67], [0.0, 0.0, 0.0], [0.5] * 3)
        res = make_midi([60, 64, 67, 65, 69, 72],
                        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.5] * 6)
        result = compare_performance_ED(res, ref)
        assert result.stats["total_chords_extra"] == 1
 
    def test_imperfect_chord_counted(self):
        # One pitch wrong in a three-note chord -> imperfect, not fully correct
        ref = make_midi([60, 64, 67], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        res = make_midi([60, 63, 67], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        result = compare_performance_ED(res, ref)
        assert result.stats["total_chords_imperfect"] == 1
        assert result.stats["total_chords_correct"] == 0
 
    def test_stats_with_mix_note_chord(self):
        # Reference: one single note then C major and F-major chords.
        # Response: missing the F-major chord. Note is correct.
        # Checks that chord stats are counted correctly even when notes are present.
        ref = make_midi([60, 60, 64, 67, 65, 69, 72],
                        [0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                        [0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        res = make_midi([60, 60, 64, 67],
                        [0.0, 1.0, 1.0, 1.0],
                        [0.4, 0.5, 0.5, 0.5])
        result = compare_performance_ED(res, ref)
        assert result.stats["total_notes_missing"] == 0
        assert result.stats["total_chords_missing"] == 1
        assert result.stats["total_chords_correct"] == 1
        assert result.stats["total_notes_in_reference"] == 1
        assert result.stats["total_chords_in_reference"] == 2


# 8. Tests for evaluation_function (Lambda Feedback integration)
# ------------------------------------------------------------------------------
class TestEvaluationFunction(unittest.TestCase):
    """    
    the core logic is already covered by TestComparePerformanceED above.
    simple checks here to ensure the interface is working as expected.
    """
    def test_perfect_performance_is_correct(self):
        # Single note (C4) then one C major chord
        midi = make_midi([60, 60, 64, 67], [0.0, 1.0, 1.0, 1.0], [0.4, 0.5, 0.5, 0.5])
        result = evaluation_function(midi, midi, {})
        assert result["is_correct"] == True
 
    def test_pitch_error_is_not_correct(self):
        ref = make_midi([60, 60, 64, 67], [0.0, 1.0, 1.0, 1.0], [0.4, 0.5, 0.5, 0.5])
        res = make_midi([60, 60, 63, 67], [0.0, 1.0, 1.0, 1.0], [0.4, 0.5, 0.5, 0.5])
        result = evaluation_function(res, ref, {})
        assert result["is_correct"] == False


# 9. Tests for parameter overrides
# ------------------------------------------------------------------------------
class TestParamOverrides(unittest.TestCase):
 
    def test_tight_timing_threshold_triggers_warning(self):
        # A very strict timing threshold should flag a note that is slightly late.
        # Note 3 is late while others are on time, so the residual is detectable.
        ref = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        res = make_midi([60, 62, 64, 65], [0, 0.5, 1.3, 1.5], [0.4] * 4)
        result = compare_performance_ED(
            res, ref, timing_relative_threshold=0.01
        )
        assert result.stats["total_notes_wrong_timing"] > 0
 
    def test_custom_gap_penalty_passed_through(self):
        # Note 2 is 3 semitones off (62 -> 65), so replacement cost = 3.
        # A deletion + insertion each cost gap_penalty once, total = 2 * gap_penalty.
        # gap_penalty=1 -> gap cost = 2*1 = 2 < 3  -> aligner prefers gap  -> missing > 0
        # gap_penalty=6 -> gap cost = 2*6 = 12 > 3 -> aligner prefers replacement -> missing == 0
        # So the two assertions together prove that gap_penalty is being passed through.
        ref = make_midi([60, 62], [0, 0.5], [0.4, 0.4])
        res = make_midi([60, 65], [0, 0.5], [0.4, 0.4])  # 3 semitones off
        result_lenient = compare_performance_ED(res, ref, gap_penalty=1)
        assert result_lenient.stats["total_notes_missing"] > 0
        result_default = compare_performance_ED(res, ref)
        assert result_default.stats["total_notes_missing"] == 0

    def test_custom_chord_onset_window_affects_grouping(self):
        # Two notes 80ms apart: grouped with 100ms window, separate with 50ms window
        ref = make_midi([60, 64], [0.00, 0.08], [0.5, 0.5])
        res = make_midi([60, 64], [0.00, 0.08], [0.5, 0.5])
        # With 100ms window: grouped as one chord -> chord_count=1
        result_wide = compare_performance_ED(res, ref, chord_onset_window=0.10)
        # With 50ms window: two separate notes -> chord_count=0
        result_narrow = compare_performance_ED(res, ref, chord_onset_window=0.05)
        assert result_wide.stats["total_chords_in_reference"] == 1
        assert result_narrow.stats["total_notes_in_reference"] == 2