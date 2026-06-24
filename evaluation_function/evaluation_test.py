"""
evaluation_tests.py
===================
Unit tests for the compareMusic evaluation function.
    Read the docs on how to use unittest here:
    https://docs.python.org/3/library/unittest.html
Run locally with:  python -m pytest evaluation_test.py -v
 
Sections
--------
1.  Helper: make_midi
2.  Tests for normalize_start_times
3.  Tests for note_alignment_ED
4.  Tests for estimate_global_timing and estimate_global_duration_scale
5.  Tests for note_level_feedback and compute_stats
6.  Tests for evaluation_function (Lambda Feedback integration)
7.  Tests for parameter overrides
"""


import unittest
import json
from .compare_MIDI import (
    normalize_start_times,
    compute_cost,
    note_alignment_ED,
    estimate_global_timing,
    estimate_global_duration_scale,
    compare_performance_ED,
    DEFAULT_GAP_PENALTY,
    TIMING_RELATIVE_THRESHOLD,
    DURATION_RELATIVE_THRESHOLD,
    GLOBAL_SLOW_THRESHOLD,
    GLOBAL_FAST_THRESHOLD,
)
from .evaluation import evaluation_function

# 1. Helper: make MIDI notes for testing
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


# 2. Tests for normalize_start_times
# ------------------------------------------------------------------------------
class TestNormalizeStartTimes:

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


# 3. Tests for note_alignment_ED
# ------------------------------------------------------------------------------
class TestNoteAlignmentED:
 
    def test_perfect_match_all_match_ops(self):
        ref = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        res = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        operations, D = note_alignment_ED(res["notes"], ref["notes"])
        types = [op["type"] for op in operations]
        assert all(t == "match" for t in types)
 
    def test_missing_note_detected(self):
        # pitch 62 missing in response
        ref = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        res = make_midi([60, 64],     [0, 1.0],      [0.4, 0.4])
        operations, D = note_alignment_ED(res["notes"], ref["notes"])
        types = [op["type"] for op in operations]
        assert "missing" in types
 
    def test_extra_note_detected(self):
        # extra pitch 62 in response
        ref = make_midi([60, 64],     [0, 1.0],      [0.4, 0.4])
        res = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        operations, D = note_alignment_ED(res["notes"], ref["notes"])
        types = [op["type"] for op in operations]
        assert "extra" in types
 
    def test_wrong_pitch_is_replacement(self):
        ref = make_midi([60, 62], [0, 0.5], [0.4, 0.4])
        res = make_midi([60, 65], [0, 0.5], [0.4, 0.4])
        operations, D = note_alignment_ED(res["notes"], ref["notes"])
        replacements = [op for op in operations if op["type"] == "replacement"]
        assert len(replacements) == 1
 
    def test_ops_are_in_forward_order(self):
        ref = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        res = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        operations, D = note_alignment_ED(res["notes"], ref["notes"])
        ref_indices = [op["reference_idx"] for op in operations if op["reference_idx"] is not None]
        assert ref_indices == sorted(ref_indices)
 
    def test_cost_matrix_shape(self):
        ref = make_midi([60, 62, 64], [0, 0.5, 1.0], [0.4, 0.4, 0.4])
        res = make_midi([60, 62],     [0, 0.5],       [0.4, 0.4])
        operations, D = note_alignment_ED(res["notes"], ref["notes"])
        assert D.shape == (len(res["notes"]) + 1, len(ref["notes"]) + 1)


# 4. Tests for estimate_global_timing and estimate_global_duration_scale
# ------------------------------------------------------------------------------
class TestGlobalEstimations:
 
    def test_perfect_timing_scale_is_one(self):
        ref = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        res = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        operations, D = note_alignment_ED(res["notes"], ref["notes"])
        scale, offset = estimate_global_timing(operations, res["notes"], ref["notes"])
        assert abs(scale - 1.0) < 0.01
 
    def test_slower_playing_scale_greater_than_one(self):
        ref = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        res = make_midi([60, 62, 64, 65], [0, 0.6, 1.2, 1.8], [0.4] * 4)  # 20% slower
        operations, D = note_alignment_ED(res["notes"], ref["notes"])
        scale, offset = estimate_global_timing(operations, res["notes"], ref["notes"])
        assert scale > 1.0
 
    def test_perfect_duration_scale_is_one(self):
        ref = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        res = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        operations, D = note_alignment_ED(res["notes"], ref["notes"])
        dur_scale = estimate_global_duration_scale(operations, res["notes"], ref["notes"])
        assert abs(dur_scale - 1.0) < 0.01
 
    def test_fewer_than_3_matched_returns_defaults(self):
        # Only 2 notes -- should return (1.0, 0.0)
        ref = make_midi([60, 62], [0, 0.5], [0.4, 0.4])
        res = make_midi([60, 62], [0, 0.5], [0.4, 0.4])
        operations, D = note_alignment_ED(res["notes"], ref["notes"])
        scale, offset = estimate_global_timing(operations, res["notes"], ref["notes"])
        dur_scale = estimate_global_duration_scale(operations, res["notes"], ref["notes"])
        assert scale == 1.0
        assert offset == 0.0
        assert dur_scale == 1.0


# 5. Tests for note_level_feedback and compute_stats
# ------------------------------------------------------------------------------
class TestComparePerformanceED:
 
    def test_consistent_tempo_not_flagged_per_note(self):
        """
        A student playing consistently 20% slower should NOT get
        note-level timing warnings -- only a global tempo comment.
        """
        ref = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        res = make_midi([60, 62, 64, 65], [0, 0.6, 1.2, 1.8], [0.4] * 4)
        result = compare_performance_ED(res, ref)
        for n in result.note_details:
            if n["operation_type"] in ("match", "replacement"):
                assert n["timing_correct"] == True
 
    def test_single_late_note_flagged(self):
        """
        One note that is very late compared to the rest should be flagged
        after the global trend is removed.
        """
        ref = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        res = make_midi([60, 62, 64, 65], [0, 0.5, 1.8, 1.5], [0.4] * 4)  # note 3 very late
        result = compare_performance_ED(res, ref)
        flagged = [n for n in result.note_details if not n["timing_correct"]]
        assert len(flagged) > 0
 
    def test_pitch_error_recorded_correctly(self):
        ref = make_midi([60, 62], [0, 0.5], [0.4, 0.4])
        res = make_midi([60, 65], [0, 0.5], [0.4, 0.4])  # 3 semitones off
        result = compare_performance_ED(res, ref)
        replacements = [n for n in result.note_details if n["operation_type"] == "replacement"]
        assert len(replacements) == 1
        assert replacements[0]["pitch_diff"] == 3
        assert replacements[0]["pitch_correct"] == False
 
    def test_all_correct_stats(self):
        midi = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        result = compare_performance_ED(midi, midi)
        assert result.stats["total_notes_missing"] == 0
        assert result.stats["total_notes_extra"] == 0
        assert result.stats["total_notes_wrong_pitch"] == 0
        assert result.stats["pitch_all_correct"] == True
 
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


# 6. Tests for evaluation_function (Lambda Feedback integration)
# ------------------------------------------------------------------------------
class TestEvaluationFunction:
    """    
    the core logic is already covered by TestComparePerformanceED above.
    simple checks here to ensure the interface is working as expected.
    """
    def test_perfect_performance_is_correct(self):
        midi = make_midi([60, 62, 64, 65], [0, 0.5, 1.0, 1.5], [0.4] * 4)
        result = evaluation_function(midi, midi, {})
        assert result["is_correct"] == True
 
    def test_pitch_error_is_not_correct(self):
        ref = make_midi([60, 62], [0, 0.5], [0.4, 0.4])
        res = make_midi([60, 65], [0, 0.5], [0.4, 0.4])
        result = evaluation_function(res, ref, {})
        assert result["is_correct"] == False


# 7. Tests for parameter overrides
# ------------------------------------------------------------------------------
class TestParamOverrides:
 
    def test_tight_timing_threshold_triggers_warning(self):
        """
        A very strict timing threshold should flag a note that is slightly late.
        Note 3 is late while others are on time, so the residual is detectable.
        """
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