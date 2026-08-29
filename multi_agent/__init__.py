"""
Multi-Agent Module for AI Autonomous Vehicle Platform.
Facilitates V2V / V2I mesh network communication and cooperative platoon management (CACC).
"""

from .v2x_network import V2XNetworkMesh
from .platoon_manager import PlatoonManager, PlatoonVehicle

__all__ = ["V2XNetworkMesh", "PlatoonManager", "PlatoonVehicle"]
