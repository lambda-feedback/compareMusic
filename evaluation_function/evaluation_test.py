import unittest
from .evaluation import Params, evaluation_function
import json

with open("./data/referenceMIDI.json") as f:
    reference = json.load(f)

with open("./data/learnerMIDI.json") as f:
    learner = json.load(f)


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

    def test_evaluation(self):
        result = evaluation_function(learner, reference, Params()).to_dict()
        self.assertFalse(result["is_correct"])
