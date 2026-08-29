import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class FusedTrack:
    track_id: int
    label: str
    x: float
    y: float
    vx: float
    vy: float
    covariance: np.ndarray

class SensorFusionPipeline:
    """
    Extended Kalman Filter / Weighted Fusion pipeline combining LiDAR, Radar, and Camera data.
    """
    def __init__(self):
        self.tracks: Dict[int, FusedTrack] = {}

    def fuse(self, camera_dets: List, lidar_points: np.ndarray, radar_targets: List) -> List[FusedTrack]:
        """
        Fuses multimodal sensor data to synthesize robust state estimates.
        """
        fused_results = []
        
        # Merge objects into single track framework
        for det in camera_dets:
            # Simple weighted average / state estimate update simulation
            lidar_dist = 0.0
            if len(lidar_points) > 0:
                lidar_dist = np.min(np.linalg.norm(lidar_points[:, :2], axis=1))

            vx, vy = det.velocity
            if radar_targets:
                r_target = radar_targets[0]
                vx = 0.7 * vx + 0.3 * r_target.get("doppler_speed", vx)

            track = FusedTrack(
                track_id=det.object_id,
                label=det.label,
                x=float(det.position[0]),
                y=float(det.position[1]),
                vx=float(vx),
                vy=float(vy),
                covariance=np.eye(4) * 0.05
            )
            fused_results.append(track)
            self.tracks[track.track_id] = track

        return fused_results
