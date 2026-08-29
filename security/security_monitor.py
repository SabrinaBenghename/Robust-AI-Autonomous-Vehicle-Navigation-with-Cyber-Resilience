import math

from security.integrity import (
    make_signed_message,
    verify_signed_message,
)

from security.security_logger import (
    SecurityLogger,
)


class SecurityMonitor:

    def __init__(
        self,
        logger=None,
    ):

        # ======================================================
        # LOGGER
        # ======================================================

        if logger is None:

            logger = (
                SecurityLogger()
            )

        self.logger = logger

        # ======================================================
        # SENSOR LIMITS
        # ======================================================

        self.sensor_minimum = 0.0

        self.sensor_maximum = 2000.0

        # Large single-frame changes can indicate spoofing.
        #
        # We intentionally keep this fairly tolerant because
        # traffic vehicles can enter/leave a sensor cone.

        self.sensor_maximum_jump = 900.0

        # ======================================================
        # LANE PREDICTION LIMITS
        # ======================================================

        self.screen_width = 1280.0

        self.expected_lane_boundaries = 5

        self.minimum_lane_spacing = 40.0

        self.maximum_lane_jump = 220.0

        # ======================================================
        # SAFE MODE
        # ======================================================

        self.safe_mode = False

        self.safe_mode_frames = 60

        self.safe_mode_remaining = 0

        self.safe_mode_reason = ""

        # ======================================================
        # TRUSTED HISTORY
        # ======================================================

        self.previous_sensor_data = {}

        self.previous_lane_predictions = None

        # ======================================================
        # STATISTICS
        # ======================================================

        self.total_checks = 0

        self.total_alerts = 0

        self.integrity_failures = 0

        self.sensor_anomalies = 0

        self.lane_anomalies = 0

        self.current_frame = 0

    # ==========================================================
    # FRAME UPDATE
    # ==========================================================

    def begin_frame(
        self,
        frame,
    ):

        self.current_frame = int(
            frame
        )

        if (
            self.safe_mode_remaining
            > 0
        ):

            self.safe_mode_remaining -= 1

        if (
            self.safe_mode_remaining
            <= 0
        ):

            self.safe_mode_remaining = 0

            self.safe_mode = False

            self.safe_mode_reason = ""

    # ==========================================================
    # ACTIVATE SAFE MODE
    # ==========================================================

    def activate_safe_mode(
        self,
        reason,
        frame=None,
    ):

        if frame is None:
            frame = self.current_frame

        self.safe_mode = True

        self.safe_mode_reason = str(
            reason
        )

        self.safe_mode_remaining = (
            self.safe_mode_frames
        )

    # ==========================================================
    # SECURITY ALERT
    # ==========================================================

    def security_alert(
        self,
        frame,
        source,
        event,
        details=None,
        severity="WARNING",
    ):

        self.total_alerts += 1

        self.activate_safe_mode(
            event,
            frame,
        )

        self.logger.log(
            frame=frame,
            source=source,
            event=event,
            severity=severity,
            details=(
                details
                if details is not None
                else {}
            ),
            action="DATA REJECTED - SAFE MODE ACTIVATED",
        )

    # ==========================================================
    # CREATE SIGNED SENSOR MESSAGE
    # ==========================================================

    def create_signed_sensor_message(
        self,
        sensor_data,
    ):

        return make_signed_message(
            sensor_data
        )

    # ==========================================================
    # VERIFY SIGNED SENSOR MESSAGE
    # ==========================================================

    def verify_sensor_message(
        self,
        message,
        frame=None,
    ):

        if frame is None:
            frame = self.current_frame

        self.total_checks += 1

        valid = verify_signed_message(
            message
        )

        if valid:

            return (
                True,
                message["payload"],
            )

        self.integrity_failures += 1

        self.security_alert(
            frame=frame,
            source="SENSOR_MESSAGE",
            event="MESSAGE_INTEGRITY_FAILURE",
            details={
                "reason": (
                    "HMAC verification failed"
                )
            },
            severity="HIGH",
        )

        return (
            False,
            None,
        )

    # ==========================================================
    # VALIDATE SENSOR DATA
    # ==========================================================

    def validate_sensor_data(
        self,
        sensor_data,
        frame=None,
    ):

        if frame is None:
            frame = self.current_frame

        self.total_checks += 1

        if not isinstance(
            sensor_data,
            dict,
        ):

            self.sensor_anomalies += 1

            self.security_alert(
                frame=frame,
                source="SENSORS",
                event="INVALID_SENSOR_MESSAGE",
                details={
                    "received_type": str(
                        type(sensor_data)
                    )
                },
                severity="HIGH",
            )

            return (
                self.previous_sensor_data.copy(),
                False,
            )

        cleaned = {}

        valid = True

        # ======================================================
        # CHECK EVERY SENSOR
        # ======================================================

        for name, value in (
            sensor_data.items()
        ):

            # --------------------------------------------------
            # NUMERIC CHECK
            # --------------------------------------------------

            try:

                numeric_value = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                valid = False

                self.sensor_anomalies += 1

                self.security_alert(
                    frame=frame,
                    source=str(name),
                    event="NON_NUMERIC_SENSOR_VALUE",
                    details={
                        "value": str(
                            value
                        )
                    },
                )

                if (
                    name
                    in self.previous_sensor_data
                ):

                    cleaned[name] = (
                        self.previous_sensor_data[
                            name
                        ]
                    )

                continue

            # --------------------------------------------------
            # FINITE CHECK
            # --------------------------------------------------

            if not math.isfinite(
                numeric_value
            ):

                valid = False

                self.sensor_anomalies += 1

                self.security_alert(
                    frame=frame,
                    source=str(name),
                    event="NON_FINITE_SENSOR_VALUE",
                    details={
                        "value": str(
                            numeric_value
                        )
                    },
                )

                if (
                    name
                    in self.previous_sensor_data
                ):

                    cleaned[name] = (
                        self.previous_sensor_data[
                            name
                        ]
                    )

                continue

            # --------------------------------------------------
            # RANGE CHECK
            # --------------------------------------------------

            if (
                numeric_value
                < self.sensor_minimum
                or
                numeric_value
                > self.sensor_maximum
            ):

                valid = False

                self.sensor_anomalies += 1

                self.security_alert(
                    frame=frame,
                    source=str(name),
                    event="SENSOR_VALUE_OUT_OF_RANGE",
                    details={
                        "value": (
                            numeric_value
                        ),

                        "minimum": (
                            self.sensor_minimum
                        ),

                        "maximum": (
                            self.sensor_maximum
                        ),
                    },
                )

                if (
                    name
                    in self.previous_sensor_data
                ):

                    cleaned[name] = (
                        self.previous_sensor_data[
                            name
                        ]
                    )

                else:

                    cleaned[name] = (
                        self.sensor_maximum
                    )

                continue

            # --------------------------------------------------
            # SUDDEN-JUMP CHECK
            # --------------------------------------------------

            if (
                name
                in self.previous_sensor_data
            ):

                previous_value = float(
                    self.previous_sensor_data[
                        name
                    ]
                )

                jump = abs(
                    numeric_value
                    - previous_value
                )

                if (
                    jump
                    > self.sensor_maximum_jump
                ):

                    valid = False

                    self.sensor_anomalies += 1

                    self.security_alert(
                        frame=frame,
                        source=str(name),
                        event="SUSPICIOUS_SENSOR_JUMP",
                        details={
                            "previous_value": (
                                previous_value
                            ),

                            "new_value": (
                                numeric_value
                            ),

                            "jump": (
                                jump
                            ),
                        },
                    )

                    cleaned[name] = (
                        previous_value
                    )

                    continue

            # --------------------------------------------------
            # TRUST VALUE
            # --------------------------------------------------

            cleaned[name] = (
                numeric_value
            )

        # ======================================================
        # SAVE TRUSTED DATA
        # ======================================================

        if valid:

            self.previous_sensor_data = (
                cleaned.copy()
            )

        else:

            # Keep valid portions while retaining previous
            # trusted values for rejected fields.

            combined = (
                self.previous_sensor_data.copy()
            )

            combined.update(
                cleaned
            )

            self.previous_sensor_data = (
                combined
            )

            cleaned = combined

        return (
            cleaned,
            valid,
        )

    # ==========================================================
    # VALIDATE AI LANE PREDICTIONS
    # ==========================================================

    def validate_lane_predictions(
        self,
        predictions,
        frame=None,
    ):

        if frame is None:
            frame = self.current_frame

        self.total_checks += 1

        # ======================================================
        # BASIC STRUCTURE
        # ======================================================

        if predictions is None:

            self.lane_anomalies += 1

            self.security_alert(
                frame=frame,
                source="ROBUST_LANENET_V3",
                event="MISSING_LANE_PREDICTIONS",
                severity="HIGH",
            )

            return (
                self.previous_lane_predictions,
                False,
            )

        try:

            values = [
                float(value)
                for value
                in predictions
            ]

        except (
            TypeError,
            ValueError,
        ):

            self.lane_anomalies += 1

            self.security_alert(
                frame=frame,
                source="ROBUST_LANENET_V3",
                event="INVALID_LANE_PREDICTIONS",
                details={
                    "predictions": str(
                        predictions
                    )
                },
                severity="HIGH",
            )

            return (
                self.previous_lane_predictions,
                False,
            )

        # ======================================================
        # EXPECT EXACTLY FIVE BOUNDARIES
        # ======================================================

        if (
            len(values)
            != self.expected_lane_boundaries
        ):

            self.lane_anomalies += 1

            self.security_alert(
                frame=frame,
                source="ROBUST_LANENET_V3",
                event="INVALID_LANE_COUNT",
                details={
                    "expected": (
                        self.expected_lane_boundaries
                    ),

                    "received": (
                        len(values)
                    ),
                },
                severity="HIGH",
            )

            return (
                self.previous_lane_predictions,
                False,
            )

        # ======================================================
        # FINITE + RANGE
        # ======================================================

        for lane_index, value in enumerate(
            values
        ):

            if not math.isfinite(
                value
            ):

                self.lane_anomalies += 1

                self.security_alert(
                    frame=frame,
                    source="ROBUST_LANENET_V3",
                    event="NON_FINITE_LANE_VALUE",
                    details={
                        "lane_index": (
                            lane_index
                        ),

                        "value": str(
                            value
                        ),
                    },
                )

                return (
                    self.previous_lane_predictions,
                    False,
                )

            if (
                value < 0.0
                or
                value > self.screen_width
            ):

                self.lane_anomalies += 1

                self.security_alert(
                    frame=frame,
                    source="ROBUST_LANENET_V3",
                    event="LANE_VALUE_OUT_OF_RANGE",
                    details={
                        "lane_index": (
                            lane_index
                        ),

                        "value": (
                            value
                        ),
                    },
                )

                return (
                    self.previous_lane_predictions,
                    False,
                )

        # ======================================================
        # ORDER + SPACING
        # ======================================================

        for lane_index in range(
            len(values) - 1
        ):

            spacing = (
                values[
                    lane_index + 1
                ]
                - values[
                    lane_index
                ]
            )

            if (
                spacing
                < self.minimum_lane_spacing
            ):

                self.lane_anomalies += 1

                self.security_alert(
                    frame=frame,
                    source="ROBUST_LANENET_V3",
                    event="INVALID_LANE_GEOMETRY",
                    details={
                        "left_index": (
                            lane_index
                        ),

                        "right_index": (
                            lane_index + 1
                        ),

                        "spacing": (
                            spacing
                        ),
                    },
                    severity="HIGH",
                )

                return (
                    self.previous_lane_predictions,
                    False,
                )

        # ======================================================
        # TEMPORAL JUMP CHECK
        # ======================================================

        if (
            self.previous_lane_predictions
            is not None
        ):

            for lane_index in range(
                len(values)
            ):

                previous_value = float(
                    self.previous_lane_predictions[
                        lane_index
                    ]
                )

                new_value = float(
                    values[
                        lane_index
                    ]
                )

                jump = abs(
                    new_value
                    - previous_value
                )

                if (
                    jump
                    > self.maximum_lane_jump
                ):

                    self.lane_anomalies += 1

                    self.security_alert(
                        frame=frame,
                        source="ROBUST_LANENET_V3",
                        event="SUSPICIOUS_LANE_JUMP",
                        details={
                            "lane_index": (
                                lane_index
                            ),

                            "previous_value": (
                                previous_value
                            ),

                            "new_value": (
                                new_value
                            ),

                            "jump": (
                                jump
                            ),
                        },
                    )

                    return (
                        self.previous_lane_predictions,
                        False,
                    )

        # ======================================================
        # TRUST PREDICTIONS
        # ======================================================

        self.previous_lane_predictions = (
            values.copy()
        )

        return (
            values,
            True,
        )

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(
        self,
    ):

        return {

            "safe_mode": (
                self.safe_mode
            ),

            "safe_mode_reason": (
                self.safe_mode_reason
            ),

            "safe_mode_remaining": (
                self.safe_mode_remaining
            ),

            "total_checks": (
                self.total_checks
            ),

            "total_alerts": (
                self.total_alerts
            ),

            "integrity_failures": (
                self.integrity_failures
            ),

            "sensor_anomalies": (
                self.sensor_anomalies
            ),

            "lane_anomalies": (
                self.lane_anomalies
            ),
        }