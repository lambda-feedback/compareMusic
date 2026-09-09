"""
backend_test.py
===============
Tests that the audio-to-MIDI transcription backend is actually available.

Basic Pitch does not ship inference code of its own. It picks a backend at
import time from whichever of CoreML, TensorFlow, TFLite or ONNX Runtime it
finds installed, and if it finds none it fails while defining
ICASSP_2022_MODEL_PATH:

    NameError: name '_default_model_type' is not defined

Basic Pitch declares those backends behind platform markers, so which one
arrives depends on the machine doing the install:

    - macOS       -> coremltools
    - Linux/amd64 -> tensorflow
    - Linux/arm64 -> the `tensorflow` wheel is an empty shim that requires
                     `tensorflow-cpu-aws`, which does not get resolved into
                     our lock, so no backend is installed at all

The third case is not hypothetical: it is every container built on an Apple
Silicon machine, where the worker dies at import and the container never
starts. CI and the AWS build are amd64, so they never see it.

The fix is to stop relying on those markers and declare a backend
explicitly. ONNX Runtime is the one with wheels for every platform we
build for, so these tests assert it is present.

Run locally with:  python -m pytest evaluation_function/backend_test.py -v
"""

import importlib


def test_onnx_runtime_is_installed():
    """
    A backend must be declared explicitly rather than inherited from a
    platform marker, so that arm64 containers get one too.
    """
    assert importlib.util.find_spec("onnxruntime") is not None, (
        "onnxruntime is not installed; basic-pitch needs an explicitly "
        "declared inference backend so that every build platform gets one"
    )


def test_basic_pitch_resolves_a_model_path():
    """
    The symptom that takes the container down. Importing this name fails
    with NameError when no backend is installed.
    """
    from basic_pitch import ICASSP_2022_MODEL_PATH

    assert ICASSP_2022_MODEL_PATH is not None


def test_model_loads_through_our_own_loader():
    """
    Guards the entry point the evaluation function actually calls, so a
    missing or broken backend fails here rather than at container start.
    """
    from .audio_processing import load_basic_pitch_model

    assert load_basic_pitch_model() is not None
