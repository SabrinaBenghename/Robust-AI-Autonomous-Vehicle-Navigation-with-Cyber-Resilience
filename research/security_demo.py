import copy
import os
import sys


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


from security.security_logger import (
    SecurityLogger,
)

from security.security_monitor import (
    SecurityMonitor,
)


# ==============================================================
# PRINT RESULT
# ==============================================================

def print_result(
    title,
    valid,
    value,
    monitor,
):

    print()
    print("-" * 72)

    print(
        title
    )

    print("-" * 72)

    print(
        "VALID:",
        valid
    )

    print(
        "VALUE:",
        value
    )

    print(
        "SAFE MODE:",
        monitor.safe_mode
    )

    if monitor.safe_mode:

        print(
            "REASON:",
            monitor.safe_mode_reason
        )


# ==============================================================
# MAIN
# ==============================================================

def main():

    logger = (
        SecurityLogger()
    )

    # Start this demo with an empty log.

    logger.clear()

    monitor = (
        SecurityMonitor(
            logger=logger
        )
    )

    print()
    print("=" * 72)
    print(
        "AUTONOMOUS VEHICLE CYBERSECURITY DEMO"
    )
    print("=" * 72)

    # ==========================================================
    # TEST 1
    # NORMAL SENSOR MESSAGE
    # ==========================================================

    monitor.begin_frame(
        1
    )

    normal_sensor_data = {

        "front": 750.0,

        "left": 1200.0,

        "right": 980.0,
    }

    signed_message = (
        monitor.create_signed_sensor_message(
            normal_sensor_data
        )
    )

    integrity_ok, payload = (
        monitor.verify_sensor_message(
            signed_message,
            frame=1,
        )
    )

    if integrity_ok:

        cleaned, valid = (
            monitor.validate_sensor_data(
                payload,
                frame=1,
            )
        )

    else:

        cleaned = None
        valid = False

    print_result(
        "TEST 1 - NORMAL AUTHENTIC SENSOR MESSAGE",
        valid,
        cleaned,
        monitor,
    )

    # ==========================================================
    # TEST 2
    # NORMAL SECOND FRAME
    # ==========================================================

    monitor.begin_frame(
        2
    )

    normal_sensor_data_2 = {

        "front": 710.0,

        "left": 1185.0,

        "right": 960.0,
    }

    cleaned, valid = (
        monitor.validate_sensor_data(
            normal_sensor_data_2,
            frame=2,
        )
    )

    print_result(
        "TEST 2 - NORMAL SENSOR VALUES",
        valid,
        cleaned,
        monitor,
    )

    # ==========================================================
    # TEST 3
    # MESSAGE TAMPERING
    # ==========================================================

    monitor.begin_frame(
        3
    )

    legitimate_message = (
        monitor.create_signed_sensor_message(
            {
                "front": 680.0,
                "left": 1150.0,
                "right": 950.0,
            }
        )
    )

    # ----------------------------------------------------------
    # SIMULATED TAMPERING
    # ----------------------------------------------------------
    #
    # The payload is modified but the original HMAC is kept.
    # Verification should therefore fail.
    # ----------------------------------------------------------

    tampered_message = copy.deepcopy(
        legitimate_message
    )

    tampered_message[
        "payload"
    ][
        "front"
    ] = 25.0

    valid, payload = (
        monitor.verify_sensor_message(
            tampered_message,
            frame=3,
        )
    )

    print_result(
        "TEST 3 - MODIFIED MESSAGE / HMAC FAILURE",
        valid,
        payload,
        monitor,
    )

    # ==========================================================
    # TEST 4
    # SENSOR SPOOFING / INVALID VALUE
    # ==========================================================

    monitor.begin_frame(
        4
    )

    spoofed_sensor_data = {

        "front": -500.0,

        "left": 1140.0,

        "right": 940.0,
    }

    cleaned, valid = (
        monitor.validate_sensor_data(
            spoofed_sensor_data,
            frame=4,
        )
    )

    print_result(
        "TEST 4 - SPOOFED FRONT SENSOR VALUE",
        valid,
        cleaned,
        monitor,
    )

    # ==========================================================
    # TEST 5
    # NORMAL AI LANE PREDICTIONS
    # ==========================================================

    monitor.begin_frame(
        5
    )

    normal_lanes = [
        342.0,
        493.0,
        639.0,
        789.0,
        937.0,
    ]

    lanes, valid = (
        monitor.validate_lane_predictions(
            normal_lanes,
            frame=5,
        )
    )

    print_result(
        "TEST 5 - NORMAL ROBUSTLANENET V3 OUTPUT",
        valid,
        lanes,
        monitor,
    )

    # ==========================================================
    # TEST 6
    # SPOOFED AI LANE OUTPUT
    # ==========================================================

    monitor.begin_frame(
        6
    )

    spoofed_lanes = [
        342.0,
        493.0,
        639.0,
        789.0,
        1700.0,
    ]

    lanes, valid = (
        monitor.validate_lane_predictions(
            spoofed_lanes,
            frame=6,
        )
    )

    print_result(
        "TEST 6 - INVALID / SPOOFED LANE PREDICTION",
        valid,
        lanes,
        monitor,
    )

    # ==========================================================
    # FINAL STATUS
    # ==========================================================

    status = (
        monitor.get_status()
    )

    print()
    print("=" * 72)
    print(
        "FINAL SECURITY STATUS"
    )
    print("=" * 72)

    print(
        "Total checks:",
        status[
            "total_checks"
        ]
    )

    print(
        "Total alerts:",
        status[
            "total_alerts"
        ]
    )

    print(
        "Integrity failures:",
        status[
            "integrity_failures"
        ]
    )

    print(
        "Sensor anomalies:",
        status[
            "sensor_anomalies"
        ]
    )

    print(
        "Lane anomalies:",
        status[
            "lane_anomalies"
        ]
    )

    print(
        "Safe mode:",
        status[
            "safe_mode"
        ]
    )

    print()

    print(
        "Security log:"
    )

    print(
        logger.output_path
    )

    print("=" * 72)
    print()


if __name__ == "__main__":

    main()