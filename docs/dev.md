# compareMusic

Automated formative feedback on music practice. Compares a student's MIDI performance against a reference MIDI and generates formatve, note-level feedback covering pitch accuracy, timing, and note duration.

## Inputs
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gap_penalty` | float | `6` | Alignment cost for a missing or extra note. Increase this if the function incorrectly splits one wrong note into a "missing + extra" pair. |
| `timing_relative_threshold` | float | `0.20` | Timing tolerance as a fraction of the inter-onset interval (IOI). `0.20` means up to 20% of the interval between consecutive notes. |
| `duration_relative_threshold` | float | `0.25` | Duration tolerance as a fraction of the reference note's duration. `0.25` means up to 25%. |
| `global_slow_threshold` | float | `1.15` | If the student's overall tempo scale exceeds this value, the overview reports "your tempo is slower than the reference". |
| `global_fast_threshold` | float | `0.85` | If the student's overall tempo scale falls below this value, the overview reports "your tempo is faster than the reference". |

Both `response` and `answer` (i.e. reference) must be a JSON object with a `notes` array:
 
```json
{
  "notes": [
    {"pitch": 60, "start": 0.00, "duration": 0.50},
    {"pitch": 62, "start": 0.60, "duration": 0.50}
  ]
}
```

where `pitch` is an integer representing MIDI note number (e.g. middle C = 60), `start` is float representing note onset time in seconds, and `duration` is float in seconds.

## Outputs

| Field | Type | Description |
|-------|------|-------------|
| `is_correct` | bool | `true` only when there are no missing notes, no extra notes, all pitches correct, all timing within threshold, and all durations within threshold |
| `feedback` | string | Human-readable feedback string |
 
The feedback string is divided into two sections:
 
**Overview** — overall tempo judgement, and counts of pitch errors, missing notes, and extra notes.
 
**Detail** — note-by-note breakdown of every specific issue found.


## Examples
 
### Perfect performance
 
```python
response = {
  "notes": [
    {"pitch": 60, "start": 0.00, "duration": 0.50},
    {"pitch": 62, "start": 0.60, "duration": 0.50}
  ]
}
answer = {
  "notes": [
    {"pitch": 60, "start": 0.00, "duration": 0.50},
    {"pitch": 62, "start": 0.60, "duration": 0.50}
  ]
}
params = {}
```
 
```python
{
  "is_correct": True,
  "feedback": "Overview: \nTiming: your overall tempo is within an acceptable range. Good job! ...\n\nGreat performance! No further issues found."
}
```
 
### Wrong pitch and missing note
 
```python
response = {
  "notes": [
    {"pitch": 60, "start": 0.00, "duration": 0.50},
    {"pitch": 63, "start": 0.60, "duration": 0.50},
    {"pitch": 64, "start": 1.35, "duration": 0.50},
    {"pitch": 65, "start": 1.80, "duration": 0.70}
  ]
}
answer = {
  "notes": [
    {"pitch": 60, "start": 0.00, "duration": 0.50},
    {"pitch": 62, "start": 0.60, "duration": 0.50},
    {"pitch": 64, "start": 1.20, "duration": 0.50},
    {"pitch": 65, "start": 1.80, "duration": 0.50},
    {"pitch": 67, "start": 2.50, "duration": 0.50}
  ]
}
params = {}
```
 
```python
{
  "is_correct": False,
  "feedback": "Overview: \nTiming: your overall tempo is within an acceptable range. ...\nThere is 1 note played with the wrong pitch.\nThere is 1 note you missed from the reference.\nThere are no extra notes. Good job!\n\nDetail: \nNote 5 (pitch 67) is missing in your performance.\nNote 2: wrong pitch -- expected 62, played 63 (1 semitone(s) off)."
}
```