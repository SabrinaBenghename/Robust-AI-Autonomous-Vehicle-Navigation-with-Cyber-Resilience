# AI Autonomous Vehicle Platform Architecture

Overview of software architecture, functional layers, and data pipelines powering the Autonomous Vehicle Platform.

```mermaid
graph TD
    Sensors[Sensors Suite: Camera/LiDAR/Radar] --> Perception[AI Perception & Sensor Fusion]
    Perception --> Planning[Behavior & Path Planner]
    Planning --> Controller[PID & Pure Pursuit Control]
    Controller --> Physics[Vehicle Dynamics & Physics Engine]
    Physics --> Screen[PyGame Simulation Display]

    DigitalTwin[Digital Twin Sync] <--> Telemetry[Vehicle CAN Telemetry]
    Cybersecurity[CAN IDS Security] <--> Telemetry
    MultiAgent[V2X Mesh Network] <--> Planning
```

## Core Modules

1. **`simulator/`**: PyGame engine, physics kinematics, road generation, traffic, pedestrians, and camera tracking.
2. **`ai/`**: Unified perception pipelines, path planning (A*), behavior state machines, and motion controllers.
3. **`digital_twin/`**: Real-time cloud sync, telemetry diagnostics, and predictive maintenance.
4. **`cybersecurity/`**: CAN bus Intrusion Detection System (IDS) and V2X message authentication.
5. **`multi_agent/`**: V2V mesh communications and cooperative platooning.
