class PIDController:
    """
    Proportional-Integral-Derivative Controller for longitudinal throttle & brake command generation.
    """
    def __init__(self, kp: float = 0.8, ki: float = 0.1, kd: float = 0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, target_value: float, current_value: float, dt: float = 0.1) -> float:
        error = target_value - current_value
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)
        return output

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
