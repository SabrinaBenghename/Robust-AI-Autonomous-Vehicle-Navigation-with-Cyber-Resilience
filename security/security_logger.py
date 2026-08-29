import json
import os
from datetime import datetime


class SecurityLogger:

    def __init__(
        self,
        output_path=None,
    ):

        # ======================================================
        # PROJECT DIRECTORY
        # ======================================================

        current_file = os.path.abspath(
            __file__
        )

        security_directory = os.path.dirname(
            current_file
        )

        project_root = os.path.dirname(
            security_directory
        )

        # ======================================================
        # DEFAULT LOG FILE
        # ======================================================

        if output_path is None:

            output_path = os.path.join(
                project_root,
                "results",
                "security",
                "security_events.jsonl",
            )

        self.output_path = (
            output_path
        )

        # ======================================================
        # CREATE DIRECTORY
        # ======================================================

        output_directory = os.path.dirname(
            self.output_path
        )

        os.makedirs(
            output_directory,
            exist_ok=True,
        )

        # ======================================================
        # IN-MEMORY EVENTS
        # ======================================================

        self.recent_events = []

        self.total_events = 0

    # ==========================================================
    # LOG EVENT
    # ==========================================================

    def log(
        self,
        frame,
        source,
        event,
        severity="WARNING",
        details=None,
        action=None,
    ):

        if details is None:
            details = {}

        entry = {

            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),

            "frame": int(
                frame
            ),

            "source": str(
                source
            ),

            "event": str(
                event
            ),

            "severity": str(
                severity
            ),

            "details": details,

            "action": action,
        }

        # ======================================================
        # WRITE JSONL
        # ======================================================

        with open(
            self.output_path,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                )
            )

            file.write(
                "\n"
            )

        # ======================================================
        # MEMORY
        # ======================================================

        self.recent_events.append(
            entry
        )

        self.total_events += 1

        # Keep memory small.

        if (
            len(
                self.recent_events
            )
            > 100
        ):

            self.recent_events = (
                self.recent_events[-100:]
            )

        return entry

    # ==========================================================
    # GET RECENT EVENTS
    # ==========================================================

    def get_recent_events(
        self,
        limit=10,
    ):

        return self.recent_events[
            -int(limit):
        ]

    # ==========================================================
    # CLEAR LOG FILE
    # ==========================================================

    def clear(
        self,
    ):

        with open(
            self.output_path,
            "w",
            encoding="utf-8",
        ):
            pass

        self.recent_events = []

        self.total_events = 0