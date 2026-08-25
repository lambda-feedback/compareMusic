"""
alignment_visualisation.py
========================
Visualisation of the event-alignment dynamic programming output
"""

import matplotlib.pyplot as plt

def format_event_pitch(event):
    """
    Convert an event into a short pitch label for the alignment diagram.
    A note event is shown as a single MIDI pitch, e.g. 60
    A chord event is shown as a set of MIDI pitches, e.g.: {60, 64, 67}
    """
    pitches = [note["pitch"] for note in event["notes"]] # list of MIDI pitches in the event

    if len(pitches) == 1:
        return str(pitches[0]) # single note

    return "{" + ", ".join(str(pitch) for pitch in pitches) + "}" # chord

def plot_edit_distance_alignment(operations, D, response_events, ref_events):
    """
    Plot:
    1. The accumulated edit-distance cost matrix and optimal path.
    2. The recovered event-level alignment diagram.

    Args:
        operations: Alignment operations returned by event_alignment_ED().
        D: Accumulated cost matrix returned by event_alignment_ED().
        response_events: Student response event sequence.
        ref_events: Reference event sequence.
    """
    N = len(response_events)
    M = len(ref_events)

    # Plot 1: accumulated cost matrix
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(D, origin="lower", cmap="Greys")

    # Display the accumulated cost in every matrix cell
    for i in range(N + 1):
        for j in range(M + 1):
            ax.text(j, i, str(int(D[i, j])), 
                    ha="center", va="center", fontsize=9)

    # Recover the coordinates of the optimal path from the operations
    n = N
    m = M
    path_rows = [n]
    path_cols = [m]

    for operation in reversed(operations):
        operation_type = operation["type"]
        if operation_type in ("match", "replacement"): # both indices decrease
            n -= 1
            m -= 1
        elif operation_type == "extra": # only the response index decreases
            n -= 1
        elif operation_type == "missing": # only the reference index decreases
            m -= 1
        path_rows.append(n)
        path_cols.append(m)

    path_rows.reverse()
    path_cols.reverse()

    # Plot the optimal path
    ax.plot(path_cols, path_rows, "ro-", 
            markersize=8, linewidth=2)
    ax.set_xticks(range(M + 1))
    ax.set_yticks(range(N + 1))
    ax.set_xlabel("Reference event index")
    ax.set_ylabel("Response event index")
    ax.set_title("Accumulated Cost Matrix D")
    plt.tight_layout()
    plt.show()

    # Plot 2: alignment diagram
    # ------------------------------------------------------------------
    box_width = 1.2
    box_height = 0.6

    reference_y = 2.5
    response_y = 0.5

    x_gap = 2.0

    reference_x_positions = {}
    response_x_positions = {}

    x_position = 0.0

    # Assign one horizontal position to each alignment operation
    for operation in operations:
        operation_type = operation["type"]

        if operation_type == "missing":
            reference_x_positions[operation["reference_idx"]] = x_position
        elif operation_type == "extra":
            response_x_positions[operation["response_idx"]] = x_position
        else:
            reference_x_positions[operation["reference_idx"]] = x_position
            response_x_positions[operation["response_idx"]] = x_position

        x_position += x_gap

    figure_width = max(10, len(operations) * 2)

    fig, ax = plt.subplots(figsize=(figure_width, 4))

    ax.axis("off")

    # Draw reference events
    for reference_idx, xpos in reference_x_positions.items():
        event = ref_events[reference_idx]
        pitch_label = format_event_pitch(event)
        ax.add_patch(
            plt.Rectangle(
                (xpos - box_width / 2, reference_y),
                box_width, box_height, fill=False,
                edgecolor="black", linewidth=1.5)
        )

        ax.text(xpos, reference_y + box_height * 0.70,
            "x" + str(reference_idx + 1),
            ha="center", va="center", fontsize=8, color="gray")

        ax.text(xpos, reference_y + box_height * 0.25,
            pitch_label, ha="center", va="center", fontsize=9)

    # Draw response events
    for response_idx, xpos in response_x_positions.items():
        event = response_events[response_idx]
        pitch_label = format_event_pitch(event)
        ax.add_patch(
                    plt.Rectangle(
                        (xpos - box_width / 2, response_y),
                        box_width, box_height, fill=False,
                        edgecolor="black", linewidth=1.5)
                )

        ax.text(xpos, response_y + box_height * 0.70,
            "y" + str(response_idx + 1),
            ha="center", va="center", fontsize=8, color="gray")

        ax.text(xpos, response_y + box_height * 0.25,
            pitch_label, ha="center", va="center", fontsize=9)

    # Draw operation arrows and labels
    middle_y = (reference_y + response_y + box_height) / 2
    for operation in operations:
        operation_type = operation["type"]
        if operation_type == "missing":
            xpos = reference_x_positions[operation["reference_idx"]]
            ax.annotate("",
                        xy=(xpos, response_y + box_height),
                        xytext=(xpos, reference_y),
                        arrowprops={"arrowstyle": "<->", "color": "gray",
                                    "linestyle": "dashed", "linewidth": 1.5})
            ax.text(xpos, middle_y, "missing", ha="center",
                    va="center", fontsize=8, color="red")

        elif operation_type == "extra":
            xpos = response_x_positions[operation["response_idx"]]
            ax.text(xpos, middle_y, "extra", ha="center", va="center",
                    fontsize=8, color="purple")
            
        elif operation_type == "match":
            xpos = reference_x_positions[operation["reference_idx"]]
            ax.annotate("", 
                        xy=(xpos, response_y + box_height),
                        xytext=(xpos, reference_y),
                        arrowprops={"arrowstyle": "<->", 
                                    "color": "green", "linewidth": 2})

        elif operation_type == "replacement":
            xpos = reference_x_positions[operation["reference_idx"]]
            ax.annotate("", 
                        xy=(xpos, response_y + box_height),
                        xytext=(xpos, reference_y),
                        arrowprops={"arrowstyle": "<->", 
                                    "color": "red", "linewidth": 2})

    ax.set_xlim(-1, max(x_position - x_gap + 1, 1))
    ax.set_ylim(0, 4)
    ax.set_title("Alignment Diagram")
    plt.tight_layout()
    plt.show()