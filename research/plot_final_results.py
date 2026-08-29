import csv
import os

import matplotlib.pyplot as plt


# ==============================================================
# PROJECT PATHS
# ==============================================================

CURRENT_FILE = os.path.abspath(__file__)
RESEARCH_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(RESEARCH_DIR)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "final_evaluation",
)

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "final_summary.csv",
)

FIGURES_DIR = os.path.join(
    RESULTS_DIR,
    "figures",
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True,
)


# ==============================================================
# LOAD CSV
# ==============================================================

def load_results():

    if not os.path.exists(CSV_PATH):

        raise FileNotFoundError(
            f"Could not find:\n{CSV_PATH}"
        )

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    if not rows:

        raise RuntimeError(
            "final_summary.csv contains no data."
        )

    return rows


# ==============================================================
# FORMAT CONDITION NAME
# ==============================================================

def format_condition(condition):

    return (
        condition
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


# ==============================================================
# SAVE FIGURE
# ==============================================================

def save_figure(filename):

    path = os.path.join(
        FIGURES_DIR,
        filename,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        path
    )


# ==============================================================
# LABEL BARS
# ==============================================================

def add_labels(
    bars,
    decimals=2,
    suffix="",
):

    for bar in bars:

        value = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            value,
            f"{value:.{decimals}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


# ==============================================================
# FIGURE 1
# PERCEPTION ERROR
# ==============================================================

def plot_perception_error(
    conditions,
    values,
):

    plt.figure(
        figsize=(9, 5.5)
    )

    bars = plt.bar(
        conditions,
        values,
    )

    plt.title(
        "RobustLaneNet V3 Perception Error Across Visual Conditions",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel(
        "Visual Condition"
    )

    plt.ylabel(
        "Average Perception Lane Error (pixels)"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    add_labels(
        bars,
        decimals=2,
        suffix=" px",
    )

    save_figure(
        "perception_error_by_condition.png"
    )


# ==============================================================
# FIGURE 2
# CLOSED-LOOP LANE ERROR
# ==============================================================

def plot_lane_tracking_error(
    conditions,
    values,
):

    plt.figure(
        figsize=(9, 5.5)
    )

    bars = plt.bar(
        conditions,
        values,
    )

    plt.title(
        "Closed-Loop Stable Lane Tracking Error",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel(
        "Visual Condition"
    )

    plt.ylabel(
        "Average Stable Lane Error (pixels)"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    add_labels(
        bars,
        decimals=2,
        suffix=" px",
    )

    save_figure(
        "lane_tracking_error_by_condition.png"
    )


# ==============================================================
# FIGURE 3
# COLLISIONS
# ==============================================================

def plot_collisions(
    conditions,
    values,
):

    plt.figure(
        figsize=(9, 5.5)
    )

    bars = plt.bar(
        conditions,
        values,
    )

    plt.title(
        "Collision Count Across Visual Conditions",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel(
        "Visual Condition"
    )

    plt.ylabel(
        "Number of Collisions"
    )

    plt.ylim(
        0,
        max(values) + 1
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    add_labels(
        bars,
        decimals=0,
    )

    save_figure(
        "collisions_by_condition.png"
    )


# ==============================================================
# FIGURE 4
# AVOIDANCE SUCCESS
# ==============================================================

def plot_avoidance_success(
    conditions,
    values,
):

    plt.figure(
        figsize=(9, 5.5)
    )

    bars = plt.bar(
        conditions,
        values,
    )

    plt.title(
        "Obstacle Avoidance Success Across Visual Conditions",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel(
        "Visual Condition"
    )

    plt.ylabel(
        "Avoidance Success Rate (%)"
    )

    plt.ylim(
        0,
        110,
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    add_labels(
        bars,
        decimals=1,
        suffix="%",
    )

    save_figure(
        "avoidance_success_by_condition.png"
    )


# ==============================================================
# FIGURE 5
# AVERAGE SPEED
# ==============================================================

def plot_average_speed(
    conditions,
    values,
):

    plt.figure(
        figsize=(9, 5.5)
    )

    bars = plt.bar(
        conditions,
        values,
    )

    plt.title(
        "Average Vehicle Speed Across Visual Conditions",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel(
        "Visual Condition"
    )

    plt.ylabel(
        "Average Vehicle Speed"
    )

    plt.ylim(
        0,
        max(values) + 1.5,
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    add_labels(
        bars,
        decimals=2,
    )

    save_figure(
        "average_speed_by_condition.png"
    )


# ==============================================================
# FIGURE 6
# PERCEPTION VS CLOSED-LOOP ERROR
# ==============================================================

def plot_error_comparison(
    conditions,
    perception_values,
    lane_values,
):

    positions = list(
        range(len(conditions))
    )

    width = 0.36

    perception_positions = [
        x - width / 2.0
        for x in positions
    ]

    lane_positions = [
        x + width / 2.0
        for x in positions
    ]

    plt.figure(
        figsize=(10, 5.8)
    )

    perception_bars = plt.bar(
        perception_positions,
        perception_values,
        width=width,
        label="Perception Error",
    )

    lane_bars = plt.bar(
        lane_positions,
        lane_values,
        width=width,
        label="Stable Lane Error",
    )

    plt.title(
        "Perception Error vs Closed-Loop Lane Tracking Error",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel(
        "Visual Condition"
    )

    plt.ylabel(
        "Error (pixels)"
    )

    plt.xticks(
        positions,
        conditions,
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    add_labels(
        perception_bars,
        decimals=2,
    )

    add_labels(
        lane_bars,
        decimals=2,
    )

    save_figure(
        "perception_vs_lane_error.png"
    )


# ==============================================================
# FIGURE 7
# OFF-ROAD FRAMES
# ==============================================================

def plot_offroad_frames(
    conditions,
    values,
):

    plt.figure(
        figsize=(9, 5.5)
    )

    bars = plt.bar(
        conditions,
        values,
    )

    plt.title(
        "Off-Road Frames Across Visual Conditions",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel(
        "Visual Condition"
    )

    plt.ylabel(
        "Off-Road Frames"
    )

    plt.ylim(
        0,
        max(
            1,
            max(values) + 1,
        )
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    add_labels(
        bars,
        decimals=0,
    )

    save_figure(
        "offroad_frames_by_condition.png"
    )


# ==============================================================
# MAIN
# ==============================================================

def main():

    print()
    print("=" * 78)

    print(
        "ROBUST AI AUTONOMOUS VEHICLE"
    )

    print(
        "FINAL RESEARCH FIGURE GENERATOR"
    )

    print("=" * 78)
    print()

    rows = load_results()

    # ==========================================================
    # EXACT CSV COLUMN NAMES
    # ==========================================================

    conditions = [
        format_condition(
            row["condition"]
        )
        for row in rows
    ]

    collisions = [
        float(
            row["collisions"]
        )
        for row in rows
    ]

    avoidance_success = [
        float(
            row[
                "avoidance_success_rate_percent"
            ]
        )
        for row in rows
    ]

    offroad_frames = [
        float(
            row["off_road_frames"]
        )
        for row in rows
    ]

    average_speed = [
        float(
            row["average_speed"]
        )
        for row in rows
    ]

    stable_lane_error = [
        float(
            row[
                "average_stable_lane_error_px"
            ]
        )
        for row in rows
    ]

    perception_error = [
        float(
            row[
                "average_perception_lane_error_px"
            ]
        )
        for row in rows
    ]

    # ==========================================================
    # PRINT VALUES
    # ==========================================================

    print(
        "Conditions:",
        conditions
    )

    print(
        "Perception errors:",
        [
            round(x, 2)
            for x in perception_error
        ]
    )

    print(
        "Stable lane errors:",
        [
            round(x, 2)
            for x in stable_lane_error
        ]
    )

    print(
        "Collisions:",
        [
            int(x)
            for x in collisions
        ]
    )

    print(
        "Avoidance success:",
        [
            round(x, 1)
            for x in avoidance_success
        ]
    )

    print(
        "Average speeds:",
        [
            round(x, 2)
            for x in average_speed
        ]
    )

    print(
        "Off-road frames:",
        [
            int(x)
            for x in offroad_frames
        ]
    )

    print()

    # ==========================================================
    # GENERATE FIGURES
    # ==========================================================

    plot_perception_error(
        conditions,
        perception_error,
    )

    plot_lane_tracking_error(
        conditions,
        stable_lane_error,
    )

    plot_collisions(
        conditions,
        collisions,
    )

    plot_avoidance_success(
        conditions,
        avoidance_success,
    )

    plot_average_speed(
        conditions,
        average_speed,
    )

    plot_error_comparison(
        conditions,
        perception_error,
        stable_lane_error,
    )

    plot_offroad_frames(
        conditions,
        offroad_frames,
    )

    print()
    print("=" * 78)

    print(
        "FIGURE GENERATION COMPLETE"
    )

    print()

    print(
        "Output directory:"
    )

    print(
        FIGURES_DIR
    )

    print("=" * 78)
    print()


if __name__ == "__main__":

    main()