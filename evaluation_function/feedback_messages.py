"""
feedback_messages.py
=====================
All feedback text shown to students, plain text only,
compare_MIDI.py should only decide which message to use.

Two groups, matching the two renderers in compare_MIDI.py:
  - summary: shown by summary_feedback(), always part of the report.
  - detail:  shown by detail_feedback(), appended only when show_detail is on.
"""


# --- summary ---
# =================================================================
# ---------- current performance summary (pitch / timing / chords) ----------
pitch_summary_messages = {
    "excellent": (
        "Great! Most notes were played correctly, you've got a good "
        "grasp of the melody."
    ),
    "good": (
        "Many notes were correct, although a few passages still need "
        "more careful practice. Try slowing down in these sections and "
        "checking each note before gradually returning to the intended "
        "tempo."
    ),
    "needs_practice": (
        "Note accuracy needs more practice. Practice each short passage "
        "at a slower tempo, check each note carefully, mind the "
        "fingering during practice. Then move on to the next passage "
        "when you feel confident with the current one."
    ),
}

timing_summary_messages = {
    "excellent": (
        "Great timing consistency between notes, you've got a good "
        "sense of onset time and rhythm!"
    ),
    "good": (
        "The spacing between notes was mostly consistent, although a "
        "few passages were less steady. Practicing these sections with "
        "a slower, regular beat may help you play each note at the "
        "right time and hence make the rhythm more steady."
    ),
    "needs_practice": (
        "Timing consistency needs more practice. You can slow down in "
        "your next practice session and listen carefully for notes "
        "that arrive too early or too late. A metronome can help you "
        "play each note at the right time and hence make the rhythm "
        "more steady."
    ),
}

chord_summary_messages = {
    "needs_practice": (
        "Simultaneous notes are often hard to play correctly at the "
        "beginning. Pay attention to the fingering and hand position "
        "when playing these chords. You may find it helpful to "
        "practice each chord separately first and make sure all "
        "required notes sound together. Then you can reconnect the "
        "chords to their surrounding sections and practice at a "
        "slower tempo carefully. "
    ),
    "excellent": "Nice! The chords were played accurately overall.",
    "good": (
        "Most chord notes were played correctly, although some chords "
        "contain missing or additional notes. It's a good idea to "
        "practice each difficult chord separately and make sure that "
        "all required notes sound together."
    ),
}

# ---------- overall tempo feedback ----------
tempo_messages = {
    "slow": (
        "Your overall tempo was slower than the reference. This is not "
        "necessarily a problem, and it is a good idea to play slowly "
        "while learning. Whatever tempo you choose, aim to keep the "
        "rhythm steady throughout the performance."
    ),
    "fast": (
        "Your overall tempo was faster than the reference. This is not "
        "necessarily a problem. Although you are confident in this "
        "piece, remember to play each note clearly and make sure the "
        "rhythm remains steady."
    ),
    "on_tempo": (
        "Well done! Your overall tempo was close to the reference. "
        "Keep up the good work! Don't forget to keep the rhythm "
        "steady throughout the performance."
    ),
}

# ---------- completeness feedback (missing / extra notes & chords) ----------
completeness_level_messages = {
    "perfect": (
        "You completed the performance without missing or adding any "
        "notes or chords. Well done!"
    ),
    "mostly_complete": (
        "The performance was mostly complete, with only a few missing "
        "or additional notes or chords. Review the affected passages "
        "slowly and check your fingering before playing them again."
    ),
    "needs_more_practice": (
        "No worries! It is common to miss or play extra notes when "
        "learning a new piece, especially difficult passages. You can "
        "slow down in your next practice and pay more attention to "
        "your fingering and hand position. "
    ),
}

# ---------- next practice focus ----------
focus_messages = {
    "pitch": "You've got a good understanding of the rhythm. ",
    "timing": "You've got a good understanding of the melody. ",
    "chords": "You've got a good understanding of the melody and the rhythm. ",
    "developing": "Good progress! ",
}

focus_advice_messages = {
    "excellent_overall": (
        "Excellent work! You already have a good understanding of the "
        "melody and the rhythm. For your next attempt, choose one "
        "short challenging section and aim to play it confidently "
        "three times in a row."
    ),
    "pitch": (
        "Let's focus on note accuracy next. Choose one short "
        "challenging passage and practice it slowly. Aim to play this "
        "phrase correctly three times in a row before increasing the "
        "tempo and moving on."
    ),
    "timing": (
        "Let's focus on timing next. Practice with a slower, steady "
        "beat, preferably using a metronome. Aim to keep the spacing "
        "between the notes even three times in a row, then gradually "
        "increase the tempo."
    ),
    "chords": (
        "Let's focus on chord accuracy next. Choose one difficult "
        "chord and adjust your hand position. Aim to make all "
        "required notes sound together correctly three times in a row."
    ),
    "no_reference": "No reference notes or chords were found to evaluate.",
}

# ---------- fixed report structure ----------
report_section_titles = {
    "summary": "Practice Summary",
    "tempo": "Tempo",
    "completeness": "Performance Completeness",
    "focus": "Main Practice Focus",
}

report_closing_message = "Keep up the good work and enjoy your music journey!"


# --- detail ---
# =================================================================
# Shown above the detail section, so students know these per-note
# claims are only as reliable as the analysis behind them.
detail_caveat_message = (
    "The per-note comments below are generated automatically and may not "
    "be accurate for every note, especially if your recording was "
    "transcribed from audio."
)

note_detail_missing_message = "Note {index} (pitch {pitch}) is missing in your performance."
note_detail_extra_message = "Extra note played: pitch {pitch} at t={time:.2f}s "
note_detail_wrong_pitch_message = (
    "Note {index}: wrong pitch — expected {expected}, played {played} "
    "({semitones} semitone(s) off)."
)
note_detail_timing_message = (
    "Note {index}: timing is off by {abs_diff:.2f}s "
    "({relative_pct:.0f}% of the expected note interval), "
    "after accounting for the overall tempo trend."
)
note_detail_duration_message = (
    "Note {index}: duration is {abs_diff:.2f}s {direction} than the reference "
    "({relative_pct:.0f}% off) after accounting for the overall duration trend."
)

chord_detail_missing_message = "Chord {index} ({chord_name}) is missing in your performance."
chord_detail_extra_message = "Extra chord played: {chord_name} at event position {index}."
chord_detail_accuracy_message = (
    "Chord {index} (expected {expected}, you played {played}): "
    "{accuracy}% accurate. "
)
chord_detail_missing_pitches_suffix = "Missing note(s): {names}. "
chord_detail_extra_pitches_suffix = "Extra note(s) played: {names}."
chord_detail_timing_message = (
    "Chord {index}: timing is off by {abs_diff:.2f}s "
    "({relative_pct:.0f}% of the expected interval)."
)

no_note_errors_message = "All melody notes played correctly!!"
no_chord_errors_message = "Great performance! No further issues on chords found."
