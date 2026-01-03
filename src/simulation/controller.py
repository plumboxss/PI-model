import numpy as np

class PController:
    def __init__(self, kp, kd=0.0, shaping_factor=1.0):
        """PD controller with optional reference shaping (low-pass on error).

        Args:
            kp: proportional gain
            kd: derivative gain on pitch rate (dtheta)
            shaping_factor: 0~1 smoothing factor for reference tracking
        """
        self.kp = kp
        self.kd = kd
        # clamp shaping factor to [0, 1] for stability
        self.shaping_factor = np.clip(shaping_factor, 0.0, 1.0)
        self._prev_shaped_error = 0.0

    def _shape_error(self, raw_error):
        # Exponential smoothing to generate a softer target trajectory
        shaped = (
            self.shaping_factor * raw_error
            + (1.0 - self.shaping_factor) * self._prev_shaped_error
        )
        self._prev_shaped_error = shaped
        return shaped

    def control(self, state):
        # state: [dz_com, dtheta, dz_us_f, dz_us_r, dx_com, z_com, theta, z_us_f, z_us_r, x_com]
        # Index 6 is theta (pitch)
        pitch = state[6]
        pitch_rate = state[1] if len(state) > 1 else 0.0

        shaped_error = self._shape_error(pitch)
        action = -(self.kp * shaped_error + self.kd * pitch_rate)
        return np.array([action])

