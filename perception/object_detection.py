import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DetectedObject:
    object_id: int
    label: str  # e.g., 'car', 'pedestrian', 'cyclist', 'obstacle'
    position: Tuple[float, float, float]  # x, y, z in vehicle coordinate frame (m)
    velocity: Tuple[float, float]        # vx, vy in m/s
    bounding_box_3d: Tuple[float, float, float] # length, width, height in m
    confidence: float

class ObjectDetector:
    """
    Simulated Perception Object Detector processing camera and point cloud sensor inputs.
    """
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.tracked_count = 0

    def process_frame(self, raw_sensor_feed: dict) -> List[DetectedObject]:
        """
        Process sensor inputs and return a list of detected objects.
        """
        detected_objects = []
        raw_objects = raw_sensor_feed.get("raw_detections", [])

        for idx, obj in enumerate(raw_objects):
            confidence = obj.get("confidence", 0.9)
            if confidence >= self.confidence_threshold:
                detected_objects.append(
                    DetectedObject(
                        object_id=obj.get("id", idx + 1),
                        label=obj.get("label", "car"),
                        position=obj.get("position", (10.0, 0.0, 0.0)),
                        velocity=obj.get("velocity", (0.0, 0.0)),
                        bounding_box_3d=obj.get("dimensions", (4.5, 1.8, 1.5)),
                        confidence=confidence
                    )
                )
        return detected_objects
