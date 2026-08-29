import csv
import json
import math
import os
import random
import sys

import numpy as np
import pygame


# ==============================================================
# PROJECT PATH
# ==============================================================

CURRENT_FILE = os.path.abspath(__file__)
RESEARCH_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(RESEARCH_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from simulator.engine import SimulatorEngine
from simulator.settings import FPS


# ==============================================================
# FINAL EXPERIMENT SETTINGS
# ==============================================================

TEST_SEED = 42

# Same duration as our previous experiments.
TEST_FRAMES = 1200

CONDITIONS = [
    "CLEAN",
    "NOISE",
    "BLUR",
    "LOW_CONTRAST",
    "OCCLUSION",
]


# ==============================================================
# OUTPUT
# ==============================================================

RESULTS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results",
    "final_evaluation",
)

SUMMARY_JSON = os.path.join(
    RESULTS_FOLDER,
    "final_summary.json",
)

SUMMARY_CSV = os.path.join(
    RESULTS_FOLDER,
    "final_summary.csv",
)


# ==============================================================
# HELPERS
# ==============================================================

def mean(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def closest_lane(
    x,
    lane_centers,
):

    return min(
        range(len(lane_centers)),
        key=lambda lane: abs(
            float(x)
            - float(lane_centers[lane])
        ),
    )


def edge_clearance(
    ego,
    obstacle,
):

    # ----------------------------------------------------------
    # Horizontal separation between vehicle rectangles
    # ----------------------------------------------------------

    dx = (
        abs(
            float(ego.x)
            - float(obstacle.x)
        )
        - (
            float(ego.width)
            + float(obstacle.width)
        )
        / 2.0
    )

    # ----------------------------------------------------------
    # Vertical separation between vehicle rectangles
    # ----------------------------------------------------------

    dy = (
        abs(
            float(ego.y)
            - float(obstacle.y)
        )
        - (
            float(ego.height)
            + float(obstacle.height)
        )
        / 2.0
    )

    dx = max(
        0.0,
        dx,
    )

    dy = max(
        0.0,
        dy,
    )

    return math.hypot(
        dx,
        dy,
    )


def set_random_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    try:

        import torch

        torch.manual_seed(seed)

    except ImportError:

        pass


# ==============================================================
# PERCEPTION ERROR
# ==============================================================

def calculate_perception_error(
    simulator,
):

    predictions = getattr(
        simulator,
        "v3_lane_predictions",
        None,
    )

    if predictions is None:
        return None

    if len(predictions) != 5:
        return None

    true_boundaries = (
        simulator.world.get_lane_positions(
            simulator.screen.get_width()
        )
    )

    if len(true_boundaries) != 5:
        return None

    scale_x = (
        simulator.screen.get_width()
        / 1280.0
    )

    predicted_boundaries = [
        float(x) * scale_x
        for x in predictions
    ]

    errors = [
        abs(
            predicted
            - truth
        )
        for predicted, truth
        in zip(
            predicted_boundaries,
            true_boundaries,
        )
    ]

    return mean(errors)


# ==============================================================
# RUN ONE CONDITION
# ==============================================================

def run_condition(
    condition,
):

    # ==========================================================
    # SAME RANDOM SCENARIO FOR EVERY CONDITION
    # ==========================================================

    set_random_seed(
        TEST_SEED
    )

    # ==========================================================
    # CREATE FRESH SIMULATOR
    # ==========================================================

    simulator = SimulatorEngine()

    simulator.autonomous_mode = True

    simulator.robustness_auto_test = False

    simulator.robustness_mode = condition

    # Disable dataset generation during final evaluation.
    simulator.dataset_interval = 10**9

    # ==========================================================
    # ROAD
    # ==========================================================

    road_left = (
        simulator.screen.get_width()
        - simulator.world.road_width
    ) / 2.0

    road_right = (
        road_left
        + simulator.world.road_width
    )

    # ==========================================================
    # METRICS
    # ==========================================================

    speeds = []

    stable_lane_errors = []

    nearest_lane_errors = []

    heading_errors = []

    steering_values = []

    steering_changes = []

    edge_clearances = []

    perception_errors = []

    lane_change_durations = []

    # ==========================================================
    # COUNTERS
    # ==========================================================

    off_road_frames = 0

    emergency_events = 0

    slow_events = 0

    perception_failure_frames = 0

    # ==========================================================
    # AVOIDANCE
    # ==========================================================

    avoidance_attempts = 0

    successful_avoidances = 0

    failed_avoidances = 0

    avoidance_active = False

    collision_at_avoidance_start = 0

    # ==========================================================
    # LANE TRANSITION TRACKING
    # ==========================================================

    previous_target_lane = (
        simulator.target_lane_index
    )

    transition_active = False

    transition_start_frame = None

    transition_settle_counter = 0

    required_settle_frames = 8

    # ==========================================================
    # OTHER STATE
    # ==========================================================

    previous_steering = 0.0

    previous_reason = ""

    collision_start = getattr(
        simulator.traffic_manager,
        "collision_count",
        0,
    )

    previous_collision_count = (
        collision_start
    )

    # ==========================================================
    # HEADER
    # ==========================================================

    print()
    print("=" * 78)
    print(
        f"FINAL EXPERIMENT — {condition}"
    )
    print("=" * 78)

    print(
        "Seed:",
        TEST_SEED,
    )

    print(
        "Frames:",
        TEST_FRAMES,
    )

    print("=" * 78)
    print()

    # ==========================================================
    # SIMULATION LOOP
    # ==========================================================

    frame = 0

    while (
        simulator.running
        and frame < TEST_FRAMES
    ):

        simulator.process_events()

        if not simulator.running:
            break

        # ======================================================
        # CLEARANCE BEFORE UPDATE
        # ======================================================
        #
        # Useful because collision handling may remove a
        # collided traffic vehicle during update().
        # ======================================================

        if simulator.traffic:

            before_clearances = [
                edge_clearance(
                    simulator.vehicle,
                    obstacle,
                )
                for obstacle
                in simulator.traffic
            ]

            if before_clearances:

                edge_clearances.append(
                    min(
                        before_clearances
                    )
                )

        # ======================================================
        # UPDATE + RENDER
        # ======================================================

        simulator.update()

        simulator.render()

        simulator.clock.tick(
            FPS
        )

        vehicle = (
            simulator.vehicle
        )

        target_lane = (
            simulator.target_lane_index
        )

        target_center = float(
            simulator.lane_centers[
                target_lane
            ]
        )

        # ======================================================
        # SPEED
        # ======================================================

        speeds.append(
            abs(
                float(vehicle.speed)
            )
        )

        # ======================================================
        # NEAREST PHYSICAL LANE ERROR
        # ======================================================

        nearest_error = min(
            abs(
                float(vehicle.x)
                - float(lane_center)
            )
            for lane_center
            in simulator.lane_centers
        )

        nearest_lane_errors.append(
            nearest_error
        )

        # ======================================================
        # TARGET LANE CHANGED
        # ======================================================

        if (
            target_lane
            != previous_target_lane
        ):

            transition_active = True

            transition_start_frame = (
                frame
            )

            transition_settle_counter = 0

        # ======================================================
        # TARGET ERROR
        # ======================================================

        target_error = abs(
            float(vehicle.x)
            - target_center
        )

        # ======================================================
        # LANE TRANSITION
        # ======================================================

        if transition_active:

            if (
                target_error <= 20.0
                and abs(
                    float(vehicle.angle)
                )
                <= 3.0
            ):

                transition_settle_counter += 1

            else:

                transition_settle_counter = 0

            if (
                transition_settle_counter
                >= required_settle_frames
            ):

                if (
                    transition_start_frame
                    is not None
                ):

                    duration = (
                        frame
                        - transition_start_frame
                    )

                    lane_change_durations.append(
                        duration
                    )

                transition_active = False

                transition_start_frame = None

                transition_settle_counter = 0

        # ======================================================
        # STABLE LANE ERROR
        # ======================================================
        #
        # Do not punish the vehicle for being between lanes
        # during an intentional lane-change maneuver.
        # ======================================================

        if not transition_active:

            stable_lane_errors.append(
                target_error
            )

        # ======================================================
        # HEADING
        # ======================================================

        heading_errors.append(
            abs(
                float(vehicle.angle)
            )
        )

        # ======================================================
        # STEERING
        # ======================================================

        steering = float(
            getattr(
                simulator.controller,
                "last_steering",
                0.0,
            )
        )

        steering_values.append(
            abs(
                steering
            )
        )

        steering_changes.append(
            abs(
                steering
                - previous_steering
            )
        )

        previous_steering = (
            steering
        )

        # ======================================================
        # ROAD BOUNDARY
        # ======================================================

        half_width = (
            float(vehicle.width)
            / 2.0
        )

        vehicle_left = (
            float(vehicle.x)
            - half_width
        )

        vehicle_right = (
            float(vehicle.x)
            + half_width
        )

        if (
            vehicle_left < road_left
            or vehicle_right > road_right
        ):

            off_road_frames += 1

        # ======================================================
        # POST-UPDATE EDGE CLEARANCE
        # ======================================================

        if simulator.traffic:

            after_clearances = [
                edge_clearance(
                    vehicle,
                    obstacle,
                )
                for obstacle
                in simulator.traffic
            ]

            if after_clearances:

                edge_clearances.append(
                    min(
                        after_clearances
                    )
                )

        # ======================================================
        # PERCEPTION ERROR
        # ======================================================

        perception_error = (
            calculate_perception_error(
                simulator
            )
        )

        if perception_error is None:

            perception_failure_frames += 1

        else:

            perception_errors.append(
                perception_error
            )

        # ======================================================
        # PLANNER REASON
        # ======================================================

        reason = str(
            getattr(
                simulator.ai,
                "last_reason",
                "",
            )
        )

        # ======================================================
        # EMERGENCY EVENT
        # ======================================================

        if (
            "EMERGENCY" in reason
            and
            "EMERGENCY"
            not in previous_reason
        ):

            emergency_events += 1

        # ======================================================
        # SLOW EVENT
        # ======================================================

        if (
            reason.startswith(
                "SLOW"
            )
            and
            not previous_reason.startswith(
                "SLOW"
            )
        ):

            slow_events += 1

        # ======================================================
        # AVOIDANCE START
        # ======================================================

        if (
            target_lane
            != simulator.cruise_lane_index
            and
            previous_target_lane
            == simulator.cruise_lane_index
        ):

            avoidance_attempts += 1

            avoidance_active = True

            collision_at_avoidance_start = (
                getattr(
                    simulator.traffic_manager,
                    "collision_count",
                    0,
                )
            )

        # ======================================================
        # COLLISION COUNT
        # ======================================================

        current_collision_count = getattr(
            simulator.traffic_manager,
            "collision_count",
            0,
        )

        # ======================================================
        # COLLISION MEANS EDGE CLEARANCE = ZERO
        # ======================================================

        if (
            current_collision_count
            > previous_collision_count
        ):

            edge_clearances.append(
                0.0
            )

            previous_collision_count = (
                current_collision_count
            )

        # ======================================================
        # AVOIDANCE FAILURE
        # ======================================================

        if avoidance_active:

            if (
                current_collision_count
                > collision_at_avoidance_start
            ):

                failed_avoidances += 1

                avoidance_active = False

        # ======================================================
        # AVOIDANCE SUCCESS
        # ======================================================

        if (
            avoidance_active
            and
            target_lane
            == simulator.cruise_lane_index
            and
            previous_target_lane
            != simulator.cruise_lane_index
        ):

            successful_avoidances += 1

            avoidance_active = False

        # ======================================================
        # PROGRESS
        # ======================================================

        if (
            frame % 200
            == 0
        ):

            print(
                f"[{condition}] "
                f"frame={frame:04d} "
                f"| speed={vehicle.speed:.2f} "
                f"| lane="
                f"{closest_lane(vehicle.x, simulator.lane_centers)} "
                f"| target={target_lane} "
                f"| collisions={current_collision_count}"
            )

        # ======================================================
        # STATE
        # ======================================================

        previous_target_lane = (
            target_lane
        )

        previous_reason = (
            reason
        )

        frame += 1

    # ==========================================================
    # FINAL COUNTS
    # ==========================================================

    collision_end = getattr(
        simulator.traffic_manager,
        "collision_count",
        0,
    )

    total_collisions = (
        collision_end
        - collision_start
    )

    unfinished_avoidances = (
        1
        if avoidance_active
        else 0
    )

    finished_avoidances = (
        successful_avoidances
        + failed_avoidances
    )

    if finished_avoidances > 0:

        avoidance_success_rate = (
            successful_avoidances
            / finished_avoidances
            * 100.0
        )

    else:

        avoidance_success_rate = (
            0.0
        )

    # ==========================================================
    # EDGE CLEARANCE
    # ==========================================================

    if edge_clearances:

        minimum_edge_clearance = min(
            edge_clearances
        )

        average_edge_clearance = mean(
            edge_clearances
        )

    else:

        minimum_edge_clearance = None

        average_edge_clearance = None

    # ==========================================================
    # RESULT OBJECT
    # ==========================================================

    result = {

        "condition": condition,

        "seed": TEST_SEED,

        "frames_requested": (
            TEST_FRAMES
        ),

        "frames_completed": (
            frame
        ),

        # =====================================================
        # SAFETY
        # =====================================================

        "collisions": (
            total_collisions
        ),

        "off_road_frames": (
            off_road_frames
        ),

        "minimum_edge_clearance_px": (
            minimum_edge_clearance
        ),

        "average_edge_clearance_px": (
            average_edge_clearance
        ),

        # =====================================================
        # AVOIDANCE
        # =====================================================

        "avoidance_attempts": (
            avoidance_attempts
        ),

        "successful_avoidances": (
            successful_avoidances
        ),

        "failed_avoidances": (
            failed_avoidances
        ),

        "unfinished_avoidances": (
            unfinished_avoidances
        ),

        "avoidance_success_rate_percent": (
            avoidance_success_rate
        ),

        "emergency_events": (
            emergency_events
        ),

        "slow_events": (
            slow_events
        ),

        # =====================================================
        # VEHICLE PERFORMANCE
        # =====================================================

        "average_speed": (
            mean(
                speeds
            )
        ),

        "maximum_speed": (
            max(
                speeds
            )
            if speeds
            else 0.0
        ),

        "average_stable_lane_error_px": (
            mean(
                stable_lane_errors
            )
        ),

        "maximum_stable_lane_error_px": (
            max(
                stable_lane_errors
            )
            if stable_lane_errors
            else 0.0
        ),

        "average_nearest_lane_error_px": (
            mean(
                nearest_lane_errors
            )
        ),

        "average_heading_error_deg": (
            mean(
                heading_errors
            )
        ),

        "average_abs_steering": (
            mean(
                steering_values
            )
        ),

        "average_steering_change": (
            mean(
                steering_changes
            )
        ),

        # =====================================================
        # LANE CHANGE
        # =====================================================

        "completed_lane_transitions": (
            len(
                lane_change_durations
            )
        ),

        "average_lane_change_duration_frames": (
            mean(
                lane_change_durations
            )
        ),

        # =====================================================
        # PERCEPTION
        # =====================================================

        "average_perception_lane_error_px": (
            mean(
                perception_errors
            )
        ),

        "maximum_perception_lane_error_px": (
            max(
                perception_errors
            )
            if perception_errors
            else 0.0
        ),

        "perception_failure_frames": (
            perception_failure_frames
        ),
    }

    # ==========================================================
    # SAVE INDIVIDUAL CONDITION
    # ==========================================================

    condition_filename = (
        condition
        .lower()
        .replace(
            " ",
            "_",
        )
    )

    condition_file = os.path.join(
        RESULTS_FOLDER,
        f"{condition_filename}.json",
    )

    with open(
        condition_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
        )

    # ==========================================================
    # PRINT CONDITION RESULT
    # ==========================================================

    print()
    print("-" * 78)

    print(
        f"{condition} COMPLETE"
    )

    print("-" * 78)

    print(
        f"Collisions: "
        f"{total_collisions}"
    )

    print(
        f"Avoidance success: "
        f"{avoidance_success_rate:.1f}%"
    )

    print(
        f"Stable lane error: "
        f"{result['average_stable_lane_error_px']:.2f}px"
    )

    print(
        f"Perception error: "
        f"{result['average_perception_lane_error_px']:.2f}px"
    )

    print(
        f"Average speed: "
        f"{result['average_speed']:.2f}"
    )

    print(
        f"Off-road frames: "
        f"{off_road_frames}"
    )

    print(
        f"Emergency events: "
        f"{emergency_events}"
    )

    if (
        minimum_edge_clearance
        is not None
    ):

        print(
            f"Minimum edge clearance: "
            f"{minimum_edge_clearance:.2f}px"
        )

    print("-" * 78)
    print()

    # ==========================================================
    # CLOSE SIMULATOR
    # ==========================================================

    simulator.running = False

    pygame.quit()

    return result


# ==============================================================
# SAVE FINAL CSV
# ==============================================================

def save_csv(
    results,
):

    fields = [

        "condition",

        "collisions",

        "avoidance_attempts",

        "successful_avoidances",

        "failed_avoidances",

        "avoidance_success_rate_percent",

        "off_road_frames",

        "average_speed",

        "average_stable_lane_error_px",

        "average_nearest_lane_error_px",

        "average_perception_lane_error_px",

        "perception_failure_frames",

        "average_heading_error_deg",

        "average_steering_change",

        "emergency_events",

        "slow_events",

        "minimum_edge_clearance_px",

        "average_lane_change_duration_frames",
    ]

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for result in results:

            row = {
                field: result.get(
                    field
                )
                for field in fields
            }

            writer.writerow(
                row
            )


# ==============================================================
# PRINT FINAL TABLE
# ==============================================================

def print_final_table(
    results,
):

    print()
    print("=" * 110)

    print(
        "FINAL ROBUST-LANENET V3 CLOSED-LOOP RESULTS"
    )

    print("=" * 110)

    header = (
        f"{'CONDITION':<16}"
        f"{'COLL':>7}"
        f"{'AVOID %':>11}"
        f"{'LANE ERR':>12}"
        f"{'PERCEPT':>12}"
        f"{'SPEED':>10}"
        f"{'OFFROAD':>10}"
        f"{'EMERG':>8}"
    )

    print(
        header
    )

    print(
        "-" * 110
    )

    for result in results:

        print(
            f"{result['condition']:<16}"
            f"{result['collisions']:>7}"
            f"{result['avoidance_success_rate_percent']:>10.1f}%"
            f"{result['average_stable_lane_error_px']:>10.2f}px"
            f"{result['average_perception_lane_error_px']:>10.2f}px"
            f"{result['average_speed']:>10.2f}"
            f"{result['off_road_frames']:>10}"
            f"{result['emergency_events']:>8}"
        )

    print(
        "=" * 110
    )


# ==============================================================
# MAIN
# ==============================================================

def main():

    os.makedirs(
        RESULTS_FOLDER,
        exist_ok=True,
    )

    print()
    print("=" * 78)
    print(
        "FINAL ROBUST-LANENET V3 EXPERIMENT"
    )
    print("=" * 78)

    print()
    print(
        "This experiment runs:"
    )

    for condition in CONDITIONS:

        print(
            "  -",
            condition,
        )

    print()

    print(
        "Same traffic seed:",
        TEST_SEED,
    )

    print(
        "Frames per condition:",
        TEST_FRAMES,
    )

    print(
        "Total simulated frames:",
        TEST_FRAMES
        * len(CONDITIONS),
    )

    print()
    print(
        "After this run, the autonomous system is frozen."
    )

    print("=" * 78)
    print()

    # ==========================================================
    # RUN ALL CONDITIONS
    # ==========================================================

    results = []

    for condition in CONDITIONS:

        result = run_condition(
            condition
        )

        results.append(
            result
        )

    # ==========================================================
    # SAVE FINAL JSON
    # ==========================================================

    final_output = {

        "experiment": (
            "RobustLaneNet V3 "
            "Closed-Loop Robustness Evaluation"
        ),

        "seed": (
            TEST_SEED
        ),

        "frames_per_condition": (
            TEST_FRAMES
        ),

        "conditions": (
            CONDITIONS
        ),

        "results": (
            results
        ),
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_output,
            file,
            indent=4,
        )

    # ==========================================================
    # SAVE CSV
    # ==========================================================

    save_csv(
        results
    )

    # ==========================================================
    # FINAL TABLE
    # ==========================================================

    print_final_table(
        results
    )

    print()
    print(
        "FINAL RESULTS SAVED:"
    )

    print(
        SUMMARY_JSON
    )

    print(
        SUMMARY_CSV
    )

    print()
    print(
        "Individual condition files:"
    )

    for condition in CONDITIONS:

        filename = (
            condition
            .lower()
            .replace(
                " ",
                "_",
            )
        )

        print(
            " -",
            os.path.join(
                RESULTS_FOLDER,
                f"{filename}.json",
            ),
        )

    print()
    print("=" * 78)
    print(
        "FINAL EXPERIMENT COMPLETE"
    )
    print(
        "NO MORE CONTROLLER TUNING."
    )
    print("=" * 78)
    print()


if __name__ == "__main__":

    main()