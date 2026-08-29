"""
Digital Twin Module for AI Autonomous Vehicle Platform.
Handles real-time telemetry streaming simulation, cloud sync, state estimation, and predictive maintenance diagnostics.
"""

from .telemetry_sync import TelemetrySynchronizer, TelemetryPacket
from .predictive_maintenance import PredictiveMaintenanceEngine, ComponentHealth

__all__ = ["TelemetrySynchronizer", "TelemetryPacket", "PredictiveMaintenanceEngine", "ComponentHealth"]
