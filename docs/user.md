# compareMusic

`compareMusic` automatically evaluates a student's MIDI music performance against a reference MIDI and returns structured, formative feedback on pitch accuracy, timing, and note duration.

## What the student sees

Feedback is returned in two parts:

**Overview** — a summary of overall tempo, and counts of pitch errors, missing notes, and extra notes.

**Detail** — a note-level feedback of every specific issue, including which notes were missed, which had the wrong pitch, and which were played noticeably early, late, or with an incorrect duration. The function separates **global tempo** (playing consistently faster or slower throughout) from **local timing errors** (a single note noticeably early or late relative to surrounding notes), which means student who plays the whole piece at 80% speed will receive one global tempo comment instead of repetitive comments on every note.

## Setting up a question

Set the **Answer** field to a JSON object representing the reference MIDI performance, e.g.:

```json
{
  "notes": [
    {"pitch": 60, "start": 0.00, "duration": 0.50},
    {"pitch": 62, "start": 0.60, "duration": 0.50},
    {"pitch": 64, "start": 1.20, "duration": 0.50}
  ]
}
```

where `pitch` is an integer representing MIDI note number (e.g. middle C = 60), `start` is float representing note onset time in seconds, and `duration` is float in seconds.

The student's **Response** must be in the same format.

## Adjusting strictness

All parameters are adjustable. If not set, the defaults below are used.

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `timing_relative_threshold` | `0.20` | How much timing deviation is acceptable, as a fraction of the gap between consecutive notes. Lower = stricter. |
| `duration_relative_threshold` | `0.25` | How much duration deviation is acceptable, as a fraction of the reference note's duration. Lower = stricter. |
| `gap_penalty` | `6` | Controls note alignment. Increase this if the function incorrectly reports a wrong note as "missing + extra". |
| `global_slow_threshold` | `1.15` | Overall tempo more than 15% slower than reference triggers a "too slow" comment. |
| `global_fast_threshold` | `0.85` | Overall tempo more than 15% faster than reference triggers a "too fast" comment. |

