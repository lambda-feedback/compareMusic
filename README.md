# compareMusic

An evaluation function for the [Lambda Feedback](https://lambdafeedback.com) platform. It compares a student's music performance against a reference performance and returns formative feedback on pitch accuracy, timing, note duration and chords.

The student's response and the reference answer can each be MIDI note data, or the path to an audio recording, which is transcribed to MIDI before the comparison.

## Deployment

[![Create Release Request](https://img.shields.io/badge/Create%20Release%20Request-blue?style=for-the-badge)](https://github.com/lambda-feedback/compareMusic/issues/new?template=release-request.yml)

## Documentation

- [docs/user.md](docs/user.md) — for teachers setting up a question: input format, what the student sees, and the parameters that control strictness.
- [docs/dev.md](docs/dev.md) — inputs, outputs and worked examples.

Both files are published to the Lambda Feedback documentation site.

## Repository structure

```bash
evaluation_function/
    compare_MIDI.py       # alignment, scoring and feedback generation
    audio_processing.py   # audio-to-MIDI transcription (Basic Pitch)
    evaluation.py         # platform entry point, thin wrapper
    preview.py            # preview entry point
    main.py               # server entry point
    *_test.py             # tests

data/                     # fixtures used by the tests
docs/                     # user and developer documentation
notebooks/                # development and evaluation notebooks, see notebooks/README.md
config.json               # evaluation function name used when deploying
```

## Working on this repository

```bash
poetry install
poetry run pytest
```

To run the function in a container:

```bash
docker build -t compare-music .
docker run --rm -p 9000:8080 compare-music
```

It then answers `POST http://localhost:9000/` with a `command: eval` header and a body of `{"response": ..., "answer": ..., "params": {}}`. `GET /health` reports readiness.
