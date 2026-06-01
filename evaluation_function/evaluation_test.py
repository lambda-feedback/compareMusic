import unittest
from .evaluation import Params, evaluation_function
import json

with open("./data/referenceMIDI.json") as f:
    reference = json.load(f)

with open("./data/learnerMIDI.json") as f:
    learner = json.load(f)

def make_midi(notes):
    return {"notes": [{"pitch": p, "start": s, "duration": d} for p, s, d in notes]}

class TestEvaluationFunction(unittest.TestCase):
    """
    TestCase Class used to test the algorithm.
    ---
    Tests are used here to check that the algorithm written
    is working as it should.

    It's best practise to write these tests first to get a
    kind of 'specification' for how your algorithm should
    work, and you should run these tests before committing
    your code to AWS.

    Read the docs on how to use unittest here:
    https://docs.python.org/3/library/unittest.html

    Use evaluation_function() to check your algorithm works
    as it should.
    """

    def test_incorrect_performance(self):
        result = evaluation_function(learner, reference, Params()).to_dict()
        self.assertFalse(result["is_correct"])
        self.assertIn("feedback", result)

    def test_correct_notes(self):
        midi = make_midi([(60, 0.0, 0.5), (62, 0.6, 0.5)])
        result = evaluation_function(midi, midi, Params()).to_dict()
        self.assertTrue(result["is_correct"])

    def test_wrong_pitch(self):
        ref = make_midi([(60, 0.0, 0.5)])
        learner = make_midi([(61, 0.0, 0.5)])
        result = evaluation_function(learner, ref, Params()).to_dict()
        self.assertFalse(result["is_correct"])
        self.assertIn("wrong", result["feedback"])

    def test_timing_out_of_tolerance(self):
        ref = make_midi([(60, 0.0, 0.5)])
        learner = make_midi([(60, 0.5, 0.5)])   # difference of 0.5s, out of tolerance
        result = evaluation_function(learner, ref, Params()).to_dict()
        self.assertFalse(result["is_correct"])
        self.assertIn("start time", result["feedback"])

    def test_timing_within_tolerance(self):
        ref = make_midi([(60, 0.0, 0.5)])
        learner = make_midi([(60, 0.05, 0.5)])  # difference of 0.05s, within tolerance
        result = evaluation_function(learner, ref, Params()).to_dict()
        self.assertTrue(result["is_correct"])

    def test_missing_note(self):
        ref = make_midi([(60, 0.0, 0.5), (62, 0.6, 0.5)])
        learner = make_midi([(60, 0.0, 0.5)])
        result = evaluation_function(learner, ref, Params()).to_dict()
        self.assertFalse(result["is_correct"])
        self.assertIn("missing", result["feedback"])

    def test_extra_note(self):
        ref = make_midi([(60, 0.0, 0.5)])
        learner = make_midi([(60, 0.0, 0.5), (64, 0.6, 0.5)])
        result = evaluation_function(learner, ref, Params()).to_dict()
        self.assertFalse(result["is_correct"])
        self.assertIn("extra", result["feedback"])