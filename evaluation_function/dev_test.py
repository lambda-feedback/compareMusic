"""
dev_test.py
===========
Tests for the command line entry point in dev.py.

This is the command the README points developers at, so it should at least
run. It was inherited from the template and never adjusted: it called a
method that does not exist on the returned value, and passed its two
arguments the wrong way round.

Run locally with:  python -m pytest evaluation_function/dev_test.py -v
"""

import json
import sys

from . import dev as dev_module
from .dev import dev
from .evaluation_test import make_midi


# Helpers
# ------------------------------------------------------------------------------
TWO_NOTES = json.dumps(make_midi([60, 62], [0.0, 0.5], [0.4, 0.4]))
THREE_NOTES = json.dumps(make_midi([60, 62, 64], [0.0, 0.5, 1.0], [0.4, 0.4, 0.4]))


# Tests
# ------------------------------------------------------------------------------
def test_prints_a_result(monkeypatch, capsys):
    """The documented invocation must run and print the outcome."""
    monkeypatch.setattr(sys, "argv", ["dev", TWO_NOTES, TWO_NOTES])

    dev()

    printed = capsys.readouterr().out
    assert "is_correct" in printed
    assert "feedback" in printed


def test_reports_a_matching_performance_as_correct(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dev", TWO_NOTES, TWO_NOTES])

    dev()

    # Printed as JSON, so the boolean is lowercase.
    assert '"is_correct": true' in capsys.readouterr().out


def test_passes_arguments_in_response_then_answer_order(monkeypatch):
    """
    The first argument is the student's response and the second is the
    reference answer, matching evaluation_function's own signature. Getting
    this backwards silently swaps "missing" and "extra" in the feedback.
    """
    seen = {}

    def spy(response, answer, params):
        seen["response"] = response
        seen["answer"] = answer
        return {"is_correct": True, "feedback": ""}

    monkeypatch.setattr(dev_module, "evaluation_function", spy)
    monkeypatch.setattr(sys, "argv", ["dev", TWO_NOTES, THREE_NOTES])

    dev()

    assert seen["response"] == TWO_NOTES
    assert seen["answer"] == THREE_NOTES


def test_usage_message_when_arguments_are_missing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dev"])

    dev()

    printed = capsys.readouterr().out
    assert "usage" in printed.lower()
    # Response first, then answer. The arguments are quoted because the
    # JSON they carry contains spaces and braces.
    assert printed.index("<response>") < printed.index("<answer>")
