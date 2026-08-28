# compareMusic

`compareMusic` automatically evaluates a student's MIDI music performance against a reference MIDI and returns structured, formative feedback on pitch accuracy, timing, and note duration.

## What the student sees

The feedback message was structured into four parts. 
- **Practice Summary**: current performance level qualitatively based on accuracy of pitch, timing and chords, followed by some practice suggestions for improvement. 
- **Tempo**: the overall tempo feedback based on timing and duration scale factors. 
- **Performance Completeness**: an overview of extra or missing notes or chords based on error rate. 
- **Main Practice Focus**: suggests a measurable goal for the next attempt with a main focus area and it is selected based on accuracy of each dimension. 

## Setting up a question

The **Answer** and **Response** can be audio file path of music recordings, or MIDI (a dict, or a JSON string of a dict) format such as:
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


## Adjusting strictness

All parameters are adjustable. If not set, the defaults below are used.

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `timing_relative_threshold` | `0.20` | How much timing deviation is acceptable, as a fraction of the gap between consecutive notes. Lower = stricter. |
| `duration_relative_threshold` | `0.25` | How much duration deviation is acceptable, as a fraction of the reference note's duration. Lower = stricter. |
| `gap_penalty` | `6` | Controls note alignment. Increase this if the function incorrectly reports a wrong note as "missing + extra". |
| `global_slow_threshold` | `1.15` | Overall tempo more than 15% slower than reference triggers a "too slow" comment. |
| `global_fast_threshold` | `0.85` | Overall tempo more than 15% faster than reference triggers a "too fast" comment. |
| `chord_onset_window` | `0.05` | How close together (in seconds) notes need to start to be treated as one chord rather than separate notes. Increase this if fast chords are being incorrectly split into separate notes; decrease it if separate fast notes are being incorrectly grouped into a chord. |
