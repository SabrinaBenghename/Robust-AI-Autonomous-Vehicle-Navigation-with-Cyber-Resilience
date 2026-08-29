import copy
import os
import sys
import time

import cv2
import numpy as np
import pygame


# ==============================================================
# PROJECT PATH
# ==============================================================

CURRENT_FILE = os.path.abspath(__file__)
SIMULATOR_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(SIMULATOR_DIR)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ==============================================================
# SIMULATOR
# ==============================================================

from simulator.vision import VisionSensor
from simulator.world import World
from simulator.vehicle import Vehicle
from simulator.camera import Camera
from simulator.sensors import SensorSuite
from simulator.traffic_manager import TrafficManager
from simulator.dataset_generator import DatasetGenerator


# ==============================================================
# AI
# ==============================================================

from ai.obstacle_avoidance import ObstacleAvoidance
from ai.controller import VehicleController
from ai.perception import RobustLanePerception


# ==============================================================
# CYBERSECURITY
# ==============================================================

from security.security_monitor import SecurityMonitor


# ==============================================================
# SETTINGS
# ==============================================================

from simulator.settings import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_TITLE,
    FPS,
)


class SimulatorEngine:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode(
            (
                WINDOW_WIDTH,
                WINDOW_HEIGHT
            )
        )

        pygame.display.set_caption(
            WINDOW_TITLE
        )

        self.clock = pygame.time.Clock()

        # ====================================================
        # WORLD
        # ====================================================

        self.world = World()

        # ====================================================
        # ROAD / LANES
        # ====================================================

        boundaries = (
            self.world.get_lane_positions(
                WINDOW_WIDTH
            )
        )

        self.lane_centers = [

            (
                boundaries[0]
                + boundaries[1]
            ) / 2.0,

            (
                boundaries[1]
                + boundaries[2]
            ) / 2.0,

            (
                boundaries[2]
                + boundaries[3]
            ) / 2.0,

            (
                boundaries[3]
                + boundaries[4]
            ) / 2.0
        ]

        self.lane_width = (
            self.world.road_width
            / 4.0
        )

        # ====================================================
        # VEHICLE
        # ====================================================

        self.vehicle = Vehicle()

        self.cruise_lane_index = 2

        self.target_lane_index = (
            self.cruise_lane_index
        )

        self.vehicle.x = (
            self.lane_centers[
                self.cruise_lane_index
            ]
        )

        # ====================================================
        # CAMERA
        # ====================================================

        self.camera = Camera(
            WINDOW_WIDTH,
            WINDOW_HEIGHT
        )

        self.camera.follow(
            self.vehicle
        )

        # ====================================================
        # TRAFFIC
        # ====================================================

        self.traffic_manager = (
            TrafficManager(
                self.world.road_width
            )
        )

        self.traffic = []

        # ====================================================
        # SENSORS
        # ====================================================

        self.sensors = (
            SensorSuite()
        )

        self.vision = (
            VisionSensor()
        )

        # ====================================================
        # AI PLANNER / CONTROLLER
        # ====================================================

        self.ai = (
            ObstacleAvoidance()
        )

        self.controller = (
            VehicleController()
        )

        # ====================================================
        # ROBUST-LANENET V3
        # ====================================================

        self.robust_lane_model = (
            RobustLanePerception()
        )

        self.v3_lane_predictions = None

        # ====================================================
        # CYBERSECURITY
        # ====================================================

        self.security_monitor = (
            SecurityMonitor()
        )

        # Traffic entering or leaving a sensor cone can create
        # legitimate large distance changes. We therefore keep
        # this real-time limit tolerant.
        #
        # Range/integrity checks still detect our attacks.

        self.security_monitor.sensor_maximum_jump = (
            1600.0
        )

        # One-shot attack waiting to be injected.
        #
        # Possible values:
        #
        # SENSOR_SPOOF
        # MESSAGE_TAMPER
        # LANE_SPOOF

        self.security_attack_pending = None

        self.last_security_attack = "NONE"

        # Security fallback status for HUD.

        self.security_data_status = "TRUSTED"

        # ====================================================
        # DATASET
        # ====================================================

        self.dataset = (
            DatasetGenerator()
        )

        self.dataset_interval = 10

        self.simulation_frame = 0

        # ====================================================
        # MODES
        # ====================================================

        self.autonomous_mode = True

        self.running = True

        # ====================================================
        # ROBUSTNESS
        # ====================================================

        self.robustness_mode = "CLEAN"

        self.robustness_auto_test = False

        self.robustness_conditions = [
            "CLEAN",
            "NOISE",
            "BLUR",
            "LOW_CONTRAST",
            "OCCLUSION"
        ]

        self.robustness_index = 0

        self.robustness_duration = 3.0

        self.robustness_timer = (
            time.time()
        )

        self.last_test_image = None

        # ====================================================
        # STARTUP
        # ====================================================

        print()
        print("=" * 74)

        print(
            "ROBUST-LANENET V3 "
            "SECURE AUTONOMOUS VEHICLE"
        )

        print("=" * 74)

        print(
            "Lane centers:",
            [
                round(
                    x,
                    1
                )
                for x
                in self.lane_centers
            ]
        )

        print(
            "Cruise lane:",
            self.cruise_lane_index
        )

        print()
        print(
            "CYBERSECURITY:"
        )

        print(
            "  HMAC message integrity        : ACTIVE"
        )

        print(
            "  Sensor anomaly detection      : ACTIVE"
        )

        print(
            "  AI lane validation            : ACTIVE"
        )

        print(
            "  Security event logging        : ACTIVE"
        )

        print(
            "  Fail-safe response            : ACTIVE"
        )

        print()
        print(
            "CYBER ATTACK DEMO CONTROLS:"
        )

        print(
            "  [6] Sensor spoofing attack"
        )

        print(
            "  [7] Message tampering / HMAC attack"
        )

        print(
            "  [8] AI lane spoofing attack"
        )

        print()

        print("=" * 74)
        print()

    # ========================================================
    # TRIGGER SECURITY ATTACK
    # ========================================================

    def trigger_security_attack(
        self,
        attack_type
    ):

        self.security_attack_pending = (
            attack_type
        )

        self.last_security_attack = (
            attack_type
        )

        print()
        print("=" * 74)

        print(
            "CYBER ATTACK REQUESTED:",
            attack_type
        )

        print(
            "Attack will be injected "
            "into the next matching data message."
        )

        print("=" * 74)
        print()

    # ========================================================
    # EVENTS
    # ========================================================

    def process_events(self):

        for event in pygame.event.get():

            if (
                event.type
                == pygame.QUIT
            ):

                self.running = False

            if (
                event.type
                != pygame.KEYDOWN
            ):

                continue

            # =================================================
            # EXIT
            # =================================================

            if (
                event.key
                == pygame.K_ESCAPE
            ):

                self.running = False

            # =================================================
            # AUTONOMOUS MODE
            # =================================================

            elif (
                event.key
                == pygame.K_a
            ):

                self.autonomous_mode = (
                    not self.autonomous_mode
                )

            # =================================================
            # DATASET FRAME
            # =================================================

            elif (
                event.key
                == pygame.K_d
            ):

                lane_data = (
                    self.get_lane_ground_truth()
                )

                self.dataset.save_frame(
                    self.screen,
                    lane_data
                )

            # =================================================
            # ROBUSTNESS CONDITIONS
            # =================================================

            elif (
                event.unicode == "1"
                or
                event.key == pygame.K_1
            ):

                self.robustness_auto_test = False

                self.set_robustness_mode(
                    "CLEAN"
                )

            elif (
                event.unicode == "2"
                or
                event.key == pygame.K_2
            ):

                self.robustness_auto_test = False

                self.set_robustness_mode(
                    "NOISE"
                )

            elif (
                event.unicode == "3"
                or
                event.key == pygame.K_3
            ):

                self.robustness_auto_test = False

                self.set_robustness_mode(
                    "BLUR"
                )

            elif (
                event.unicode == "4"
                or
                event.key == pygame.K_4
            ):

                self.robustness_auto_test = False

                self.set_robustness_mode(
                    "LOW_CONTRAST"
                )

            elif (
                event.unicode == "5"
                or
                event.key == pygame.K_5
            ):

                self.robustness_auto_test = False

                self.set_robustness_mode(
                    "OCCLUSION"
                )

            # =================================================
            # CYBER ATTACK 1
            # SENSOR SPOOFING
            # =================================================

            elif (
                event.unicode == "6"
                or
                event.key == pygame.K_6
            ):

                self.trigger_security_attack(
                    "SENSOR_SPOOF"
                )

            # =================================================
            # CYBER ATTACK 2
            # MESSAGE TAMPERING
            # =================================================

            elif (
                event.unicode == "7"
                or
                event.key == pygame.K_7
            ):

                self.trigger_security_attack(
                    "MESSAGE_TAMPER"
                )

            # =================================================
            # CYBER ATTACK 3
            # AI LANE SPOOFING
            # =================================================

            elif (
                event.unicode == "8"
                or
                event.key == pygame.K_8
            ):

                self.trigger_security_attack(
                    "LANE_SPOOF"
                )

            # =================================================
            # AUTO ROBUSTNESS TEST
            # =================================================

            elif (
                event.key
                == pygame.K_t
            ):

                self.robustness_auto_test = (
                    not self.robustness_auto_test
                )

                self.robustness_index = 0

                self.robustness_timer = (
                    time.time()
                )

                self.set_robustness_mode(
                    "CLEAN"
                )

    # ========================================================
    # ROBUSTNESS MODE
    # ========================================================

    def set_robustness_mode(
        self,
        mode
    ):

        self.robustness_mode = (
            mode
        )

        self.robustness_timer = (
            time.time()
        )

        print(
            "ROBUSTNESS MODE:",
            mode
        )

    # ========================================================
    # CORRUPT IMAGE
    # ========================================================

    def apply_robustness_test(
        self,
        image
    ):

        if image is None:

            return None

        # ====================================================
        # CLEAN
        # ====================================================

        if (
            self.robustness_mode
            == "CLEAN"
        ):

            return image.copy()

        # ====================================================
        # NOISE
        # ====================================================

        if (
            self.robustness_mode
            == "NOISE"
        ):

            image_float = (
                image.astype(
                    np.float32
                )
            )

            noise = np.random.normal(
                0,
                25,
                image_float.shape
            )

            result = (
                image_float
                + noise
            )

            result = np.clip(
                result,
                0,
                255
            )

            return result.astype(
                np.uint8
            )

        # ====================================================
        # BLUR
        # ====================================================

        if (
            self.robustness_mode
            == "BLUR"
        ):

            return cv2.GaussianBlur(
                image,
                (
                    21,
                    21
                ),
                0
            )

        # ====================================================
        # LOW CONTRAST
        # ====================================================

        if (
            self.robustness_mode
            == "LOW_CONTRAST"
        ):

            result = (
                image.astype(
                    np.float32
                )
                * 0.45
                + 70
            )

            result = np.clip(
                result,
                0,
                255
            )

            return result.astype(
                np.uint8
            )

        # ====================================================
        # OCCLUSION
        # ====================================================

        if (
            self.robustness_mode
            == "OCCLUSION"
        ):

            result = (
                image.copy()
            )

            height, width = (
                result.shape[:2]
            )

            cv2.rectangle(
                result,
                (
                    int(
                        width
                        * 0.35
                    ),
                    int(
                        height
                        * 0.45
                    )
                ),
                (
                    int(
                        width
                        * 0.65
                    ),
                    int(
                        height
                        * 0.75
                    )
                ),
                (
                    0,
                    0,
                    0
                ),
                -1
            )

            return result

        return image.copy()

    # ========================================================
    # AUTO ROBUSTNESS
    # ========================================================

    def update_robustness_test(
        self
    ):

        if not self.robustness_auto_test:

            return

        if (
            time.time()
            - self.robustness_timer
            < self.robustness_duration
        ):

            return

        self.robustness_index += 1

        if (
            self.robustness_index
            >= len(
                self.robustness_conditions
            )
        ):

            self.robustness_index = 0

        self.set_robustness_mode(
            self.robustness_conditions[
                self.robustness_index
            ]
        )

    # ========================================================
    # SAFE SENSOR FALLBACK
    # ========================================================

    def get_safe_sensor_fallback(
        self,
        original_data
    ):

        previous = (
            self.security_monitor
            .previous_sensor_data
        )

        if previous:

            return (
                previous.copy()
            )

        fallback = {}

        if isinstance(
            original_data,
            dict
        ):

            for key in (
                original_data.keys()
            ):

                fallback[key] = 1500.0

        # Ensure planner always has these.

        if (
            "front"
            not in fallback
        ):

            fallback[
                "front"
            ] = 1500.0

        if (
            "left"
            not in fallback
        ):

            fallback[
                "left"
            ] = 1500.0

        if (
            "right"
            not in fallback
        ):

            fallback[
                "right"
            ] = 1500.0

        return fallback

    # ========================================================
    # SECURE SENSOR PIPELINE
    # ========================================================

    def secure_sensor_data(
        self,
        raw_sensor_data
    ):

        if not isinstance(
            raw_sensor_data,
            dict
        ):

            cleaned, valid = (
                self.security_monitor
                .validate_sensor_data(
                    raw_sensor_data,
                    frame=(
                        self.simulation_frame
                    )
                )
            )

            self.security_data_status = (
                "TRUSTED"
                if valid
                else "REJECTED"
            )

            return cleaned

        # ====================================================
        # COPY SENSOR MESSAGE
        # ====================================================

        sensor_payload = (
            raw_sensor_data.copy()
        )

        # ====================================================
        # ATTACK 1:
        # SENSOR SPOOFING
        #
        # Fake data is injected before the message is signed.
        #
        # HMAC is therefore valid, but anomaly detection must
        # reject the physically impossible value.
        # ====================================================

        if (
            self.security_attack_pending
            == "SENSOR_SPOOF"
        ):

            original_front = (
                sensor_payload.get(
                    "front",
                    None
                )
            )

            sensor_payload[
                "front"
            ] = -500.0

            self.security_attack_pending = (
                None
            )

            print()
            print("=" * 74)
            print(
                "CYBER ATTACK INJECTED"
            )
            print("=" * 74)

            print(
                "TYPE: SENSOR SPOOFING"
            )

            print(
                "SOURCE: FRONT SENSOR"
            )

            print(
                "REAL VALUE:",
                original_front
            )

            print(
                "SPOOFED VALUE:",
                sensor_payload[
                    "front"
                ]
            )

            print("=" * 74)
            print()

        # ====================================================
        # CREATE AUTHENTICATED MESSAGE
        # ====================================================

        signed_message = (
            self.security_monitor
            .create_signed_sensor_message(
                sensor_payload
            )
        )

        # ====================================================
        # ATTACK 2:
        # MESSAGE TAMPERING
        #
        # Message is signed first.
        # Then attacker changes the payload while keeping the
        # original HMAC.
        # ====================================================

        if (
            self.security_attack_pending
            == "MESSAGE_TAMPER"
        ):

            original_front = (
                signed_message[
                    "payload"
                ].get(
                    "front",
                    None
                )
            )

            tampered_message = (
                copy.deepcopy(
                    signed_message
                )
            )

            tampered_message[
                "payload"
            ][
                "front"
            ] = 25.0

            signed_message = (
                tampered_message
            )

            self.security_attack_pending = (
                None
            )

            print()
            print("=" * 74)
            print(
                "CYBER ATTACK INJECTED"
            )
            print("=" * 74)

            print(
                "TYPE: MESSAGE TAMPERING"
            )

            print(
                "PROTECTION: HMAC-SHA256"
            )

            print(
                "ORIGINAL FRONT:",
                original_front
            )

            print(
                "TAMPERED FRONT:",
                25.0
            )

            print("=" * 74)
            print()

        # ====================================================
        # VERIFY HMAC
        # ====================================================

        integrity_ok, verified_payload = (
            self.security_monitor
            .verify_sensor_message(
                signed_message,
                frame=(
                    self.simulation_frame
                )
            )
        )

        # ====================================================
        # HMAC FAILURE
        # ====================================================

        if not integrity_ok:

            self.security_data_status = (
                "REJECTED - INTEGRITY"
            )

            print()
            print(
                "SECURITY: MESSAGE REJECTED "
                "- HMAC VERIFICATION FAILED"
            )
            print()

            return (
                self.get_safe_sensor_fallback(
                    raw_sensor_data
                )
            )

        # ====================================================
        # ANOMALY VALIDATION
        # ====================================================

        cleaned_data, sensor_valid = (
            self.security_monitor
            .validate_sensor_data(
                verified_payload,
                frame=(
                    self.simulation_frame
                )
            )
        )

        if sensor_valid:

            self.security_data_status = (
                "TRUSTED"
            )

        else:

            self.security_data_status = (
                "REJECTED - ANOMALY"
            )

            print()
            print(
                "SECURITY: SUSPICIOUS SENSOR "
                "DATA REJECTED"
            )
            print()

        return cleaned_data

    # ========================================================
    # SECURE LANE PREDICTIONS
    # ========================================================

    def secure_lane_predictions(
        self,
        raw_predictions
    ):

        predictions = (
            None
            if raw_predictions is None
            else list(
                raw_predictions
            )
        )

        # ====================================================
        # ATTACK 3:
        # AI LANE SPOOFING
        # ====================================================

        if (
            self.security_attack_pending
            == "LANE_SPOOF"
        ):

            if predictions is None:

                predictions = [
                    342.0,
                    493.0,
                    639.0,
                    789.0,
                    937.0
                ]

            original = (
                predictions.copy()
            )

            if len(
                predictions
            ) >= 5:

                predictions[
                    4
                ] = 1700.0

            self.security_attack_pending = (
                None
            )

            print()
            print("=" * 74)

            print(
                "CYBER ATTACK INJECTED"
            )

            print("=" * 74)

            print(
                "TYPE: AI LANE SPOOFING"
            )

            print(
                "ORIGINAL:",
                [
                    round(
                        float(x),
                        1
                    )
                    for x
                    in original
                ]
            )

            print(
                "SPOOFED:",
                [
                    round(
                        float(x),
                        1
                    )
                    for x
                    in predictions
                ]
            )

            print("=" * 74)
            print()

        # ====================================================
        # SECURITY VALIDATION
        # ====================================================

        trusted_predictions, valid = (
            self.security_monitor
            .validate_lane_predictions(
                predictions,
                frame=(
                    self.simulation_frame
                )
            )
        )

        if valid:

            return (
                trusted_predictions
            )

        self.security_data_status = (
            "REJECTED - AI LANE"
        )

        print()
        print(
            "SECURITY: INVALID AI LANE "
            "DATA REJECTED"
        )
        print()

        # SecurityMonitor already returns the previous
        # trusted prediction if one exists.

        return (
            trusted_predictions
        )

    # ========================================================
    # V3 LANE TARGET
    # ========================================================

    def get_target_lane_x(
        self
    ):

        fallback = (
            self.lane_centers[
                self.target_lane_index
            ]
        )

        if (
            self.v3_lane_predictions
            is None
        ):

            return fallback

        if (
            len(
                self.v3_lane_predictions
            )
            != 5
        ):

            return fallback

        if (
            self.simulation_frame
            < 2
        ):

            return fallback

        scale_x = (
            self.screen.get_width()
            / 1280.0
        )

        boundaries = [

            float(x)
            * scale_x

            for x
            in self.v3_lane_predictions
        ]

        i = (
            self.target_lane_index
        )

        return (
            boundaries[i]
            + boundaries[i + 1]
        ) / 2.0

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self
    ):

        # ====================================================
        # SECURITY FRAME
        # ====================================================

        self.security_monitor.begin_frame(
            self.simulation_frame
        )

        self.update_robustness_test()

        # ====================================================
        # TRAFFIC
        # ====================================================

        self.traffic_manager.update(
            self.vehicle
        )

        self.traffic = (
            self.traffic_manager
            .get_vehicles()
        )

        # ====================================================
        # CAMERA
        # ====================================================

        camera_image = (
            self.vision.capture(
                self.screen
            )
        )

        self.vision.detect_lanes()

        # ====================================================
        # ROBUSTNESS CORRUPTION
        # ====================================================

        test_image = (
            self.apply_robustness_test(
                camera_image
            )
        )

        self.last_test_image = (
            test_image
        )

        # ====================================================
        # ROBUST-LANENET V3
        # ====================================================

        if (
            test_image
            is not None
        ):

            raw_lane_predictions = (
                self.robust_lane_model
                .predict(
                    test_image
                )
            )

            # =================================================
            # CYBERSECURITY:
            # VALIDATE AI OUTPUT BEFORE USING IT
            # =================================================

            self.v3_lane_predictions = (
                self.secure_lane_predictions(
                    raw_lane_predictions
                )
            )

        # ====================================================
        # AUTONOMOUS MODE
        # ====================================================

        if self.autonomous_mode:

            # =================================================
            # RAW SENSOR DATA
            # =================================================

            self.sensors.update(
                self.vehicle,
                self.traffic
            )

            raw_sensor_data = (
                self.sensors.get_data()
            )

            # =================================================
            # CYBERSECURITY SENSOR PIPELINE
            #
            # RAW SENSOR
            #     ↓
            # HMAC
            #     ↓
            # INTEGRITY CHECK
            #     ↓
            # ANOMALY CHECK
            #     ↓
            # TRUSTED DATA
            # =================================================

            sensor_data = (
                self.secure_sensor_data(
                    raw_sensor_data
                )
            )

            # =================================================
            # EXISTING LANE-AWARE PLANNER
            #
            # IMPORTANT:
            # Planner algorithm itself is NOT changed.
            # =================================================

            plan = self.ai.plan(
                sensor_data=(
                    sensor_data
                ),
                vehicle=(
                    self.vehicle
                ),
                traffic=(
                    self.traffic
                ),
                lane_centers=(
                    self.lane_centers
                ),
                target_lane=(
                    self.target_lane_index
                ),
                cruise_lane=(
                    self.cruise_lane_index
                ),
                lane_width=(
                    self.lane_width
                )
            )

            decision = (
                plan[
                    "decision"
                ]
            )

            self.target_lane_index = (
                plan[
                    "target_lane"
                ]
            )

            # =================================================
            # TARGET FROM TRUSTED ROBUSTLANENET V3 DATA
            # =================================================

            target_x = (
                self.get_target_lane_x()
            )

            # =================================================
            # SECURITY FAIL-SAFE RESPONSE
            #
            # We do not override an emergency STOP.
            #
            # Any other command is slowed while the security
            # system remains in safe mode.
            # =================================================

            if (
                self.security_monitor.safe_mode
                and
                decision != "STOP"
            ):

                decision = "SLOW"

            # =================================================
            # EXISTING CONTROLLER
            #
            # Controller algorithm itself is NOT changed.
            # =================================================

            control = (
                self.controller.control(
                    decision,
                    self.vehicle,
                    target_x=(
                        target_x
                    )
                )
            )

            # =================================================
            # VEHICLE
            # =================================================

            self.vehicle.apply_control(
                control[
                    "throttle"
                ],
                control[
                    "brake"
                ],
                control[
                    "steering"
                ]
            )

            # =================================================
            # COLLISION FAILSAFE
            # =================================================

            self.traffic_manager.check_collisions(
                self.vehicle
            )

            # =================================================
            # DEBUG
            # =================================================

            if (
                self.simulation_frame
                % 10
                == 0
            ):

                security_status = (
                    self.security_monitor
                    .get_status()
                )

                print()
                print(
                    f"FRONT="
                    f"{plan['front_distance']:.1f}"
                )

                print(
                    f"PLAN="
                    f"{decision} "
                    f"| TARGET LANE="
                    f"{self.target_lane_index}"
                )

                print(
                    "REASON:",
                    plan[
                        "reason"
                    ]
                )

                print(
                    f"CAR X="
                    f"{self.vehicle.x:.1f} "
                    f"| TARGET X="
                    f"{target_x:.1f} "
                    f"| SPEED="
                    f"{self.vehicle.speed:.1f}"
                )

                print(
                    f"HEADING="
                    f"{self.vehicle.angle:.2f} "
                    f"| STEER="
                    f"{control['steering']:.3f}"
                )

                print(
                    "SECURITY=",
                    (
                        "SAFE MODE"
                        if security_status[
                            "safe_mode"
                        ]
                        else "SECURE"
                    ),
                    "| ALERTS=",
                    security_status[
                        "total_alerts"
                    ]
                )

                if (
                    security_status[
                        "safe_mode"
                    ]
                ):

                    print(
                        "SECURITY REASON:",
                        security_status[
                            "safe_mode_reason"
                        ]
                    )

                print()

        else:

            self.vehicle.update_manual()

        # ====================================================
        # CAMERA FOLLOW
        # ====================================================

        self.camera.follow(
            self.vehicle
        )

        self.simulation_frame += 1

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    def get_lane_ground_truth(
        self
    ):

        return (
            self.world
            .get_lane_ground_truth(
                self.screen.get_width(),
                self.screen.get_height()
            )
        )

    # ========================================================
    # DRAW V3
    # ========================================================

    def draw_v3_lanes(
        self
    ):

        if (
            self.v3_lane_predictions
            is None
        ):

            return

        scale_x = (
            self.screen.get_width()
            / 1280.0
        )

        for lane_x in (
            self.v3_lane_predictions
        ):

            x = int(
                float(
                    lane_x
                )
                * scale_x
            )

            if (
                0
                <= x
                < self.screen.get_width()
            ):

                pygame.draw.line(
                    self.screen,
                    (
                        0,
                        255,
                        0
                    ),
                    (
                        x,
                        0
                    ),
                    (
                        x,
                        self.screen.get_height()
                    ),
                    2
                )

    # ========================================================
    # HUD
    # ========================================================

    def draw_status(
        self
    ):

        font = (
            pygame.font.Font(
                None,
                27
            )
        )

        security_status = (
            self.security_monitor
            .get_status()
        )

        if (
            security_status[
                "safe_mode"
            ]
        ):

            security_text = (
                "SAFE MODE"
            )

        else:

            security_text = (
                "SECURE"
            )

        lines = [

            (
                "Robustness: "
                + self.robustness_mode
            ),

            (
                "Cruise lane: "
                + str(
                    self.cruise_lane_index
                )
                + " | Target lane: "
                + str(
                    self.target_lane_index
                )
            ),

            (
                "Planner: "
                + self.ai.last_reason
            ),

            (
                "Security: "
                + security_text
                + " | Alerts: "
                + str(
                    security_status[
                        "total_alerts"
                    ]
                )
            ),

            (
                "Cyber demo: "
                "[6] Sensor  "
                "[7] HMAC  "
                "[8] AI Lane"
            )
        ]

        y = 20

        for line in lines:

            surface = (
                font.render(
                    line,
                    True,
                    (
                        255,
                        255,
                        255
                    )
                )
            )

            self.screen.blit(
                surface,
                (
                    20,
                    y
                )
            )

            y += 30

    # ========================================================
    # SECURITY ALERT HUD
    # ========================================================

    def draw_security_alert(
        self
    ):

        status = (
            self.security_monitor
            .get_status()
        )

        if not status[
            "safe_mode"
        ]:

            return

        # ====================================================
        # BOX
        # ====================================================

        box_x = 20
        box_y = 175
        box_width = 450
        box_height = 92

        pygame.draw.rect(
            self.screen,
            (
                120,
                0,
                0
            ),
            (
                box_x,
                box_y,
                box_width,
                box_height
            )
        )

        pygame.draw.rect(
            self.screen,
            (
                255,
                255,
                255
            ),
            (
                box_x,
                box_y,
                box_width,
                box_height
            ),
            2
        )

        # ====================================================
        # TEXT
        # ====================================================

        title_font = (
            pygame.font.Font(
                None,
                30
            )
        )

        text_font = (
            pygame.font.Font(
                None,
                23
            )
        )

        title = (
            title_font.render(
                "SECURITY ALERT - SAFE MODE ACTIVE",
                True,
                (
                    255,
                    255,
                    255
                )
            )
        )

        reason = (
            text_font.render(
                (
                    "Reason: "
                    + str(
                        status[
                            "safe_mode_reason"
                        ]
                    )
                ),
                True,
                (
                    255,
                    255,
                    255
                )
            )
        )

        action = (
            text_font.render(
                "Untrusted data rejected - vehicle speed reduced",
                True,
                (
                    255,
                    255,
                    255
                )
            )
        )

        self.screen.blit(
            title,
            (
                box_x + 10,
                box_y + 8
            )
        )

        self.screen.blit(
            reason,
            (
                box_x + 10,
                box_y + 39
            )
        )

        self.screen.blit(
            action,
            (
                box_x + 10,
                box_y + 63
            )
        )

    # ========================================================
    # V3 INPUT PREVIEW
    # ========================================================

    def draw_test_image(
        self
    ):

        if (
            self.last_test_image
            is None
        ):

            return

        preview_width = 320

        preview_height = 180

        preview = cv2.resize(
            self.last_test_image,
            (
                preview_width,
                preview_height
            )
        )

        preview = cv2.cvtColor(
            preview,
            cv2.COLOR_BGR2RGB
        )

        surface = (
            pygame.surfarray
            .make_surface(
                np.transpose(
                    preview,
                    (
                        1,
                        0,
                        2
                    )
                )
            )
        )

        x = (
            self.screen.get_width()
            - preview_width
            - 20
        )

        y = 20

        pygame.draw.rect(
            self.screen,
            (
                255,
                255,
                255
            ),
            (
                x - 2,
                y - 2,
                preview_width + 4,
                preview_height + 4
            ),
            2
        )

        self.screen.blit(
            surface,
            (
                x,
                y
            )
        )

        font = (
            pygame.font.Font(
                None,
                20
            )
        )

        label = (
            font.render(
                "V3 INPUT",
                True,
                (
                    255,
                    255,
                    255
                )
            )
        )

        self.screen.blit(
            label,
            (
                x,
                y
                + preview_height
                + 5
            )
        )

    # ========================================================
    # RENDER
    # ========================================================

    def render(
        self
    ):

        # ====================================================
        # WORLD
        # ====================================================

        self.world.draw(
            self.screen,
            self.camera
        )

        # ====================================================
        # TRAFFIC
        # ====================================================

        for traffic_vehicle in (
            self.traffic
        ):

            traffic_vehicle.draw(
                self.screen,
                self.camera
            )

        # ====================================================
        # EGO VEHICLE
        # ====================================================

        self.vehicle.draw(
            self.screen,
            self.camera
        )

        # ====================================================
        # AI LANE VISUALIZATION
        # ====================================================

        self.draw_v3_lanes()

        # ====================================================
        # HUD
        # ====================================================

        self.draw_status()

        self.draw_security_alert()

        self.draw_test_image()

        pygame.display.flip()

        # ====================================================
        # DATASET
        # ====================================================

        if (
            self.simulation_frame
            % self.dataset_interval
            == 0
        ):

            lane_data = (
                self.get_lane_ground_truth()
            )

            self.dataset.save_frame(
                self.screen,
                lane_data
            )

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self
    ):

        while self.running:

            self.process_events()

            self.update()

            self.render()

            self.clock.tick(
                FPS
            )

        pygame.quit()


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":

    SimulatorEngine().run()