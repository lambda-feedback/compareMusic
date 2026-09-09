"""
dev.py
======
Command line entry point, for trying the evaluation function without
running the server.

Usage:
    python -m evaluation_function.dev '<response>' '<answer>'

Both arguments are JSON, in the same shape the platform sends, for example:
    '{"notes": [{"pitch": 60, "start": 0.0, "duration": 0.5}]}'
"""

import json
import sys

from lf_toolkit.shared.params import Params

from .evaluation import evaluation_function

USAGE = "Usage: python -m evaluation_function.dev '<response>' '<answer>'"


def dev():
    """Run the evaluation function once and print the result."""
    if len(sys.argv) < 3:
        print(USAGE)
        return

    # Argument order matches evaluation_function itself: the student's
    # response first, the reference answer second.
    response = sys.argv[1]
    answer = sys.argv[2]

    result = evaluation_function(response, answer, Params())

    # evaluation_function returns a plain dict, so print it as JSON rather
    # than calling a serialisation method it does not have.
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    dev()
