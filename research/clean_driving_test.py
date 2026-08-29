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

CURRENT_FILE = os.path.abspath(
    __file__
)

RESEARCH_DIR = os.path.dirname(
    CURRENT_FILE
)

PROJECT_ROOT = os.path.dirname(
    RESEARCH_DIR
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from simulator.engine import SimulatorEngine
from simulator.settings import FPS


# ==============================================================
# EXPERIMENT
# ==============================================================

TEST_SEED = 42

TEST_FRAMES = 1200


RESULTS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results",
    "driving_evaluation"
)


RESULT_FILE = os.path.join(
    RESULTS_FOLDER,
    "clean_driving_diagnostic.json"
)


# ==============================================================
# HELPERS
# ==============================================================

def mean(values):

    if not values:

        return 0.0

    return (
        sum(values)
        / len(values)
    )


def safe_number(value):

    if math.isinf(value):

        return None

    if math.isnan(value):

        return None

    return float(value)


# ==============================================================
# EDGE CLEARANCE
# ==============================================================

def edge_clearance(
    ego,
    obstacle
):

    dx = (
        abs(
            float(ego.x)
            - float(obstacle.x)
        )
        - (
            float(ego.width)
            + float(obstacle.width)
        ) / 2.0
    )

    dy = (
        abs(
            float(ego.y)
            - float(obstacle.y)
        )
        - (
            float(ego.height)
            + float(obstacle.height)
        ) / 2.0
    )

    dx = max(
        0.0,
        dx
    )

    dy = max(
        0.0,
        dy
    )

    return math.hypot(
        dx,
        dy
    )


# ==============================================================
# CLOSEST LANE
# ==============================================================

def closest_lane(
    x,
    lane_centers
):

    return min(
        range(
            len(lane_centers)
        ),
        key=lambda lane: abs(
            float(x)
            - float(
                lane_centers[
                    lane
                ]
            )
        )
    )


# ==============================================================
# FORMAT VALUE
# ==============================================================

def format_value(value):

    if value is None:

        return "CLEAR"

    try:

        if math.isinf(
            float(value)
        ):

            return "CLEAR"

    except Exception:

        return str(value)

    return f"{float(value):.1f}"


# ==============================================================
# COLLISION DIAGNOSTIC
# ==============================================================

def build_collision_diagnostic(
    simulator,
    frame
):

    vehicle = (
        simulator.vehicle
    )

    ego_lane = closest_lane(
        vehicle.x,
        simulator.lane_centers
    )

    target_lane = (
        simulator.target_lane_index
    )

    reason = str(
        getattr(
            simulator.ai,
            "last_reason",
            "UNKNOWN"
        )
    )

    # ==========================================================
    # LANE INFORMATION
    # ==========================================================

    lane_gaps = getattr(
        simulator.ai,
        "last_lane_gaps",
        {}
    )

    lane_ttc = getattr(
        simulator.ai,
        "last_lane_ttc",
        {}
    )

    lanes = {}

    for lane_index in range(
        len(
            simulator.lane_centers
        )
    ):

        gaps = lane_gaps.get(
            lane_index,
            {}
        )

        ahead = gaps.get(
            "ahead",
            float("inf")
        )

        behind = gaps.get(
            "behind",
            float("inf")
        )

        ttc = lane_ttc.get(
            lane_index,
            float("inf")
        )

        lanes[
            str(lane_index)
        ] = {

            "ahead_gap_px": (
                safe_number(
                    ahead
                )
            ),

            "behind_gap_px": (
                safe_number(
                    behind
                )
            ),

            "ahead_ttc_frames": (
                safe_number(
                    ttc
                )
            )
        }

    # ==========================================================
    # NEARBY TRAFFIC
    # ==========================================================

    nearby = []

    for obstacle in simulator.traffic:

        dx = (
            float(obstacle.x)
            - float(vehicle.x)
        )

        dy = (
            float(obstacle.y)
            - float(vehicle.y)
        )

        center_distance = math.hypot(
            dx,
            dy
        )

        clearance = edge_clearance(
            vehicle,
            obstacle
        )

        obstacle_lane = closest_lane(
            obstacle.x,
            simulator.lane_centers
        )

        nearby.append({

            "lane": int(
                obstacle_lane
            ),

            "x": float(
                obstacle.x
            ),

            "y": float(
                obstacle.y
            ),

            "speed": float(
                getattr(
                    obstacle,
                    "speed",
                    0.0
                )
            ),

            "oncoming": bool(
                getattr(
                    obstacle,
                    "oncoming",
                    True
                )
            ),

            "dx_px": float(
                dx
            ),

            "dy_px": float(
                dy
            ),

            "center_distance_px": float(
                center_distance
            ),

            "edge_clearance_px": float(
                clearance
            )
        })

    nearby.sort(
        key=lambda item: item[
            "center_distance_px"
        ]
    )

    nearby = nearby[:7]

    # ==========================================================
    # SENSORS
    # ==========================================================

    sensor_data = {}

    try:

        sensor_data = (
            simulator.sensors.get_data()
        )

    except Exception:

        pass

    diagnostic = {

        "frame": int(
            frame
        ),

        "ego": {

            "x": float(
                vehicle.x
            ),

            "y": float(
                vehicle.y
            ),

            "speed": float(
                vehicle.speed
            ),

            "heading_deg": float(
                vehicle.angle
            ),

            "physical_lane": int(
                ego_lane
            ),

            "target_lane": int(
                target_lane
            )
        },

        "planner_reason": (
            reason
        ),

        "sensors": {

            key: float(value)

            for key, value
            in sensor_data.items()
        },

        "lanes": lanes,

        "nearby_traffic": (
            nearby
        )
    }

    return diagnostic


# ==============================================================
# PRINT COLLISION DIAGNOSTIC
# ==============================================================

def print_collision_diagnostic(
    diagnostic
):

    print()
    print("#" * 76)
    print(
        "COLLISION DIAGNOSTIC"
    )
    print("#" * 76)

    print(
        "Frame:",
        diagnostic[
            "frame"
        ]
    )

    ego = (
        diagnostic[
            "ego"
        ]
    )

    print(
        f"Ego X: {ego['x']:.1f}"
        f" | Speed: {ego['speed']:.2f}"
        f" | Heading: {ego['heading_deg']:.2f}°"
    )

    print(
        "Physical lane:",
        ego[
            "physical_lane"
        ],
        "| Target lane:",
        ego[
            "target_lane"
        ]
    )

    print(
        "Planner:",
        diagnostic[
            "planner_reason"
        ]
    )

    print()

    sensors = (
        diagnostic[
            "sensors"
        ]
    )

    if sensors:

        print(
            "SENSORS"
        )

        print(
            "-" * 76
        )

        for name, value in (
            sensors.items()
        ):

            print(
                f"{name.upper():>8}: "
                f"{value:.1f}px"
            )

    print()
    print(
        "LANE STATE"
    )
    print(
        "-" * 76
    )

    for lane_index, state in (
        diagnostic[
            "lanes"
        ].items()
    ):

        print(
            f"Lane {lane_index}: "
            f"ahead="
            f"{format_value(state['ahead_gap_px'])} "
            f"| TTC="
            f"{format_value(state['ahead_ttc_frames'])} "
            f"| behind="
            f"{format_value(state['behind_gap_px'])}"
        )

    print()
    print(
        "NEAREST TRAFFIC"
    )
    print(
        "-" * 76
    )

    nearby = (
        diagnostic[
            "nearby_traffic"
        ]
    )

    if not nearby:

        print(
            "No traffic detected."
        )

    else:

        for index, obstacle in enumerate(
            nearby,
            start=1
        ):

            print(
                f"{index}. "
                f"lane={obstacle['lane']} "
                f"| dx={obstacle['dx_px']:.1f} "
                f"| dy={obstacle['dy_px']:.1f} "
                f"| center={obstacle['center_distance_px']:.1f} "
                f"| edge={obstacle['edge_clearance_px']:.1f} "
                f"| speed={obstacle['speed']:.2f}"
            )

    print("#" * 76)
    print()


# ==============================================================
# MAIN
# ==============================================================

def main():

    # ==========================================================
    # REPRODUCIBILITY
    # ==========================================================

    random.seed(
        TEST_SEED
    )

    np.random.seed(
        TEST_SEED
    )

    try:

        import torch

        torch.manual_seed(
            TEST_SEED
        )

    except ImportError:

        pass

    # ==========================================================
    # RESULTS DIRECTORY
    # ==========================================================

    os.makedirs(
        RESULTS_FOLDER,
        exist_ok=True
    )

    # ==========================================================
    # SIMULATOR
    # ==========================================================

    simulator = (
        SimulatorEngine()
    )

    simulator.robustness_mode = (
        "CLEAN"
    )

    simulator.robustness_auto_test = (
        False
    )

    simulator.autonomous_mode = (
        True
    )

    # Disable automatic dataset generation.
    simulator.dataset_interval = (
        10**9
    )

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

    lane_change_durations = []

    off_road_frames = 0

    emergency_events = 0

    slow_events = 0

    # ==========================================================
    # AVOIDANCE
    # ==========================================================

    avoidance_attempts = 0

    successful_avoidances = 0

    failed_avoidances = 0

    avoidance_active = False

    collision_at_avoidance_start = 0

    # ==========================================================
    # TRANSITIONS
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
        0
    )

    previous_collision_count = (
        collision_start
    )

    # ==========================================================
    # COLLISION LOG
    # ==========================================================

    collision_diagnostics = []

    # ==========================================================
    # HEADER
    # ==========================================================

    print()
    print("=" * 76)
    print(
        "CLEAN CLOSED-LOOP COLLISION DIAGNOSTIC"
    )
    print("=" * 76)

    print(
        "Seed:",
        TEST_SEED
    )

    print(
        "Frames:",
        TEST_FRAMES
    )

    print(
        "Any collision will print a complete planner snapshot."
    )

    print("=" * 76)
    print()

    # ==========================================================
    # LOOP
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
        # SIMULATION
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
        # NEAREST LANE ERROR
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
        # TARGET CHANGE
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

            print(
                f"[FRAME {frame}] "
                f"TARGET "
                f"{previous_target_lane}"
                f" -> "
                f"{target_lane}"
            )

        # ======================================================
        # TRANSITION
        # ======================================================

        target_error = abs(
            float(vehicle.x)
            - target_center
        )

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

                    print(
                        f"[FRAME {frame}] "
                        f"LANE SETTLED "
                        f"({duration} frames)"
                    )

                transition_active = False

                transition_start_frame = None

                transition_settle_counter = 0

        # ======================================================
        # STABLE LANE ERROR
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
                0.0
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
        # ROAD
        # ======================================================

        half_width = (
            float(vehicle.width)
            / 2.0
        )

        if (
            float(vehicle.x)
            - half_width
            < road_left
        ):

            off_road_frames += 1

        elif (
            float(vehicle.x)
            + half_width
            > road_right
        ):

            off_road_frames += 1

        # ======================================================
        # CLEARANCE
        # ======================================================

        if simulator.traffic:

            frame_clearances = [

                edge_clearance(
                    vehicle,
                    obstacle
                )

                for obstacle
                in simulator.traffic
            ]

            if frame_clearances:

                edge_clearances.append(
                    min(
                        frame_clearances
                    )
                )

        # ======================================================
        # PLANNER EVENTS
        # ======================================================

        reason = str(
            getattr(
                simulator.ai,
                "last_reason",
                ""
            )
        )

        if (
            "EMERGENCY" in reason
            and
            "EMERGENCY"
            not in previous_reason
        ):

            emergency_events += 1

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
                    0
                )
            )

            print(
                f"[FRAME {frame}] "
                f"AVOIDANCE START"
            )

        # ======================================================
        # NEW COLLISION?
        # ======================================================

        current_collision_count = (
            getattr(
                simulator.traffic_manager,
                "collision_count",
                0
            )
        )

        if (
            current_collision_count
            > previous_collision_count
        ):

            new_collisions = (
                current_collision_count
                - previous_collision_count
            )

            for _ in range(
                new_collisions
            ):

                diagnostic = (
                    build_collision_diagnostic(
                        simulator,
                        frame
                    )
                )

                collision_diagnostics.append(
                    diagnostic
                )

                print_collision_diagnostic(
                    diagnostic
                )

            previous_collision_count = (
                current_collision_count
            )

        # ======================================================
        # AVOIDANCE FAILED?
        # ======================================================

        if avoidance_active:

            if (
                current_collision_count
                > collision_at_avoidance_start
            ):

                failed_avoidances += 1

                avoidance_active = False

                print(
                    f"[FRAME {frame}] "
                    f"AVOIDANCE FAILURE"
                )

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

            print(
                f"[FRAME {frame}] "
                f"AVOIDANCE SUCCESS"
            )

        # ======================================================
        # DEBUG
        # ======================================================

        if (
            frame % 100
            == 0
        ):

            print(
                f"[{frame:04d}] "
                f"speed={vehicle.speed:.2f} "
                f"| lane={closest_lane(vehicle.x, simulator.lane_centers)} "
                f"| target={target_lane} "
                f"| nearest_error={nearest_error:.1f}px "
                f"| collisions={current_collision_count} "
                f"| {reason}"
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
    # RESULTS
    # ==========================================================

    collision_end = getattr(
        simulator.traffic_manager,
        "collision_count",
        0
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

    if edge_clearances:

        minimum_edge_clearance = min(
            edge_clearances
        )

    else:

        minimum_edge_clearance = None

    # ==========================================================
    # SUMMARY
    # ==========================================================

    results = {

        "condition": "CLEAN",

        "seed": TEST_SEED,

        "frames_completed": (
            frame
        ),

        "collisions": (
            total_collisions
        ),

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

        "off_road_frames": (
            off_road_frames
        ),

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

        "average_steering_change": (
            mean(
                steering_changes
            )
        ),

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

        "minimum_edge_clearance_px": (
            minimum_edge_clearance
        ),

        # =====================================================
        # IMPORTANT
        # =====================================================

        "collision_diagnostics": (
            collision_diagnostics
        )
    }

    # ==========================================================
    # SAVE
    # ==========================================================

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # ==========================================================
    # PRINT FINAL
    # ==========================================================

    print()
    print("=" * 76)
    print(
        "CLEAN COLLISION DIAGNOSTIC RESULTS"
    )
    print("=" * 76)

    print(
        "Frames:",
        frame
    )

    print(
        "Collisions:",
        total_collisions
    )

    print(
        "Avoidance attempts:",
        avoidance_attempts
    )

    print(
        "Successful avoidances:",
        successful_avoidances
    )

    print(
        "Failed avoidances:",
        failed_avoidances
    )

    print(
        f"Avoidance success: "
        f"{avoidance_success_rate:.1f}%"
    )

    print(
        "Emergency events:",
        emergency_events
    )

    print(
        "Slow events:",
        slow_events
    )

    print(
        "Off-road frames:",
        off_road_frames
    )

    print(
        f"Average speed: "
        f"{mean(speeds):.2f}"
    )

    print(
        f"Stable lane error: "
        f"{mean(stable_lane_errors):.2f}px"
    )

    print(
        f"Nearest lane error: "
        f"{mean(nearest_lane_errors):.2f}px"
    )

    print(
        f"Average heading error: "
        f"{mean(heading_errors):.2f}°"
    )

    print(
        f"Average steering change: "
        f"{mean(steering_changes):.4f}"
    )

    print(
        f"Completed lane transitions: "
        f"{len(lane_change_durations)}"
    )

    print(
        f"Average lane-change duration: "
        f"{mean(lane_change_durations):.1f} frames"
    )

    if (
        minimum_edge_clearance
        is not None
    ):

        print(
            f"Minimum EDGE clearance: "
            f"{minimum_edge_clearance:.2f}px"
        )

    print()
    print(
        "Collision diagnostic snapshots:",
        len(
            collision_diagnostics
        )
    )

    print()
    print(
        "Saved:"
    )

    print(
        RESULT_FILE
    )

    print("=" * 76)
    print()

    pygame.quit()


if __name__ == "__main__":

    main()