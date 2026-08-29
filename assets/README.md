# Assets & Configuration Directory

This directory contains static assets, vehicle dynamics parameters, sensor specs, map configurations, and runtime presets for the AI Autonomous Vehicle Platform.

## Contents
- `config.yaml`: Primary platform parameters including PID gains, vehicle wheelbase, sensor ranges, security rules, and multi-agent platooning settings.

## Configuration Parameters
- **Vehicle Dynamics**: Wheelbase, maximum steering angle, acceleration and deceleration limits.
- **Sensors**: Field of view (FOV), maximum detection ranges for LiDAR, Radar, and Cameras.
- **Controller**: Proportional-Integral-Derivative (PID) gains for speed tracking and lookahead distance constants for Pure Pursuit steering.
- **Cybersecurity**: Intrusion Detection System (IDS) z-score thresholds and cryptosystem defaults.
