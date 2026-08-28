# compareMusic

Automated formative feedback on music practice. Compares a student's performance against a reference and generates formatve feedback.

## Inputs
| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `timing_relative_threshold` | `0.20` | How much timing deviation is acceptable, as a fraction of the gap between consecutive notes. Lower = stricter. |
| `duration_relative_threshold` | `0.25` | How much duration deviation is acceptable, as a fraction of the reference note's duration. Lower = stricter. |
| `gap_penalty` | `6` | Controls note alignment. Increase this if the function incorrectly reports a wrong note as "missing + extra". |
| `global_slow_threshold` | `1.15` | Overall tempo more than 15% slower than reference triggers a "too slow" comment. |
| `global_fast_threshold` | `0.85` | Overall tempo more than 15% faster than reference triggers a "too fast" comment. |
| `chord_onset_window` | `0.05` | How close together (in seconds) notes need to start to be treated as one chord rather than separate notes. Increase this if fast chords are being incorrectly split into separate notes; decrease it if separate fast notes are being incorrectly grouped into a chord. |

Both `response` and `answer` (i.e. reference) can be audio file path of music recordings, or MIDI (a dict, or a JSON string of a dict), they will be converted to the following format to pass through the pipeline:
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


## Outputs

| Field | Type | Description |
|-------|------|-------------|
| `is_correct` | bool | `true` only when there are no missing notes, no extra notes, all pitches correct, all timing within threshold, and all durations within threshold |
| `feedback` | string | Human-readable feedback string |
 
The feedback message was structured into four parts. 
- **Practice Summary**: current performance level qualitatively based on accuracy of pitch, timing and chords, followed by some practice suggestions for improvement. 
- **Tempo**: the overall tempo feedback based on timing and duration scale factors. 
- **Performance Completeness**: an overview of extra or missing notes or chords based on error rate. 
- **Main Practice Focus**: suggests a measurable goal for the next attempt with a main focus area and it is selected based on accuracy of each dimension. 


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
  "feedback": "Practice Summary\nGreat! Most notes were played correctly, you've got a "
              "good grasp of the melody.\nGreat timing consistency between notes, you've "
              "got a good sense of onset time and rhythm!\n\nTempo\nWell done! Your overall "
              "tempo was close to the reference. Keep up the good work! Don't forget to keep "
              "the rhythm steady throughout the performance.\n\nPerformance Completeness\n"
              "You completed the performance without missing or adding any notes or chords. "
              "Well done!\n\nMain Practice Focus\nExcellent work! You already have a good "
              "understanding of the melody and the rhythm. For your next attempt, choose one "
              "short challenging section and aim to play it confidently three times in a "
              "row.\n\nKeep up the good work and enjoy your music journey!"
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
  "feedback": "Practice Summary\nNote accuracy needs more practice. Practice each short "
              "passage at a slower tempo, check each note carefully, mind the fingering "
              "during practice. Then move on to the next passage when you feel confident "
              "with the current one.\nThe spacing between notes was mostly consistent, "
              "although a few passages were less steady. Practicing these sections with a "
              "slower, regular beat may help you play each note at the right time and hence "
              "make the rhythm more steady.\n\nTempo\nWell done! Your overall tempo was close "
              "to the reference. Keep up the good work! Don't forget to keep the rhythm "
              "steady throughout the performance.\n\nPerformance Completeness\nNo worries! "
              "It is common to miss or play extra notes when learning a new piece, especially "
              "difficult passages. You can slow down in your next practice and pay more "
              "attention to your fingering and hand position. \n\nMain Practice Focus\nGood "
              "progress! Let's focus on note accuracy next. Choose one short challenging "
              "passage and practice it slowly. Aim to play this phrase correctly three times "
              "in a row before increasing the tempo and moving on.\n\nKeep up the good work "
              "and enjoy your music journey!"
}
```