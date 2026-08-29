"""
Perception Module for AI Autonomous Vehicle Platform.
Handles object detection, lane tracking, and multi-sensor fusion (Camera + LiDAR + Radar).
"""

from .object_detection import ObjectDetector, DetectedObject
from .sensor_fusion import SensorFusionPipeline, FusedTrack
from .lane_detection import LaneDetector, LaneBoundary

__all__ = ["ObjectDetector", "DetectedObject", "SensorFusionPipeline", "FusedTrack", "LaneDetector", "LaneBoundary"]
