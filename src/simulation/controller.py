import numpy as np

class PController:
    def __init__(self, kp):
        self.kp = kp

    def control(self, state):
        # state: [dz_com, dtheta, dz_us_f, dz_us_r, dx_com, z_com, theta, z_us_f, z_us_r, x_com]
        # Index 6 is theta (pitch)
        error = state[6] 
        action = -self.kp * error
        return np.array([action])

