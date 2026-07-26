"""
pianovam_loading.py
===================
downloading the dataset, loading metadata and ground truth, 
computing mir_eval-based transcription metrics, 
"""


import json
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download


# 1. Dataset download and metadata loading
# -----------------------------------------------------------------------
def download_pianovam_dataset(data_root, repo_id="PianoVAM/PianoVAM_v1"):
    """
    Download the PianoVAM dataset into data_root if it is not already
    there. Only downloads Audio, MIDI, TSV, metadata.json, and README,
    skipping anything else in the repo.
    """
    required_paths = [
        data_root / "Audio",
        data_root / "MIDI",
        data_root / "TSV",
        data_root / "metadata.json",
    ]

    if all(path.exists() for path in required_paths):
        print("PianoVAM dataset already exists.")
        return

    print("Downloading PianoVAM dataset...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=data_root,
        allow_patterns=[
            "Audio/**",
            "MIDI/**",
            "TSV/**",
            "metadata.json",
            "README.md",
        ],
    )
    print("Download complete.")


def load_pianovam_metadata(metadata_path):
    """
    Load PianoVAM metadata.json into a DataFrame, one row per sample,
    with duration converted to seconds for convenience.
    """
    with metadata_path.open("r") as file:
        metadata = json.load(file)

    metadata_df = pd.DataFrame.from_dict(metadata, orient="index").reset_index(names="sample_id")
    metadata_df["sample_id"] = metadata_df["sample_id"].astype(str)
    metadata_df["duration_seconds"] = pd.to_timedelta(metadata_df["duration"]).dt.total_seconds()

    return metadata_df


def get_sample_paths(sample, audio_dir, midi_dir, tsv_dir):
    """
    Build the audio/MIDI/TSV file paths for one PianoVAM sample.
    """
    file_stem = str(sample["record_time"])
    return {
            "audio": audio_dir / f"{file_stem}.wav",
            "midi": midi_dir / f"{file_stem}.mid",
            "tsv": tsv_dir / f"{file_stem}.tsv",
        }


# 2. Ground truth loading and mir_eval-based metrics
# -----------------------------------------------------------------------
def load_ground_truth_notes(tsv_path):
    """
    Load PianoVAM ground truth notes from a TSV annotation file.
    """
    tsv_df = pd.read_csv(tsv_path, sep="\t")
    notes = []

    for _, row in tsv_df.iterrows():
        onset = float(row["# onset"])
        key_offset = float(row["key_offset"])
        frame_offset = float(row["frame_offset"])
        notes.append({
            "pitch": int(row["note"]),
            "onset": onset,
            "key_offset": key_offset,
            "frame_offset": frame_offset,
            "key_duration": key_offset - onset,
            "audible_duration": frame_offset - onset,
            "velocity": int(row["velocity"]),
        })

    notes.sort(key=lambda note: (note["onset"], note["pitch"]))
    return notes


def notes_to_mir_eval_arrays(notes, offset_key="offset"):
    """
    Convert a list of note dictionaries into the (intervals,
    frequencies) arrays expected by mir_eval.
    """
    intervals = np.array(
        [[note["onset"], note[offset_key]] for note in notes],
        dtype=float,
    )

    midi_pitches = np.array(
        [note["pitch"] for note in notes],
        dtype=float,
    )

    frequencies = 440.0 * (2.0 ** ((midi_pitches - 69.0) / 12.0))

    return intervals, frequencies
