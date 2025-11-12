import numpy as np
import gymnasium

from src.controller import HumanController
from src.env.components import Bump, compile_vehicle_model
from configs import Environment_Parameters, Vehicle_Parameters

STATE_KEYS = ["dz_com", "dtheta", "dz_us_f", "dz_us_r", "dx_com", 
              "z_com", "theta", "z_us_f", "z_us_r", "x_com"]
ACCEL_KEYS = ["ddz_com", "ddtheta", "ddz_us_f", "ddz_us_r", "ddx_com"]
JERK_KEYS = ["dddz_com", "dddtheta", "dddz_us_f", "dddz_us_r", "dddx_com"]

class SuspensionEnv(gymnasium.Env):
    def __init__(self, human_controller=None, is_multi_bump=False):
        super().__init__()
        self.is_multi_bump = is_multi_bump

        self.obs_keys = ["theta", "dtheta", "ddtheta", "dx_com", "ddx_com", "dz_com"]

        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.obs_keys),), dtype=np.float32)

        self.human_controller = human_controller if human_controller is not None else HumanController()
        self.config = Environment_Parameters()
        self.vehicle_params = Vehicle_Parameters()
        self.bump = Bump(is_multi_bump=self.is_multi_bump)
        self.vehicle = compile_vehicle_model(self.vehicle_params)

        self.x0 = np.array(self.config.x0, dtype=np.float32)
        self._reset_state()
        self.dt = self.config.dt_inner

        self.time = 0.0
        self.max_time = 10.0

        self.eride_time = 0.0
        self.eride_decay = 1.0
        self.eride_duration = 3.0
        self.bump_detected = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.time = 0.0
        self.eride_time = 0.0
        self.bump_detected = False

        self.bump.set_rng(self.np_random)
        self.bump.reset()

        v_ref = self.np_random.uniform(20/3.6, 40/3.6)
        self.human_controller.v_ref = v_ref
        self.x0[STATE_KEYS.index("dx_com")] = v_ref
        self._reset_state()

        info = self._get_info(z=np.zeros(2, dtype=np.float32), reward=0.0, u_eride=0.0, u_human=0.0)
        return self.obs.copy(), info

    def _reset_state(self):
        self.state = {key: 0.0 for key in STATE_KEYS}
        for i, key in enumerate(STATE_KEYS):
            self.state[key] = self.x0[i]

        self.state_ddot = {key: 0.0 for key in ACCEL_KEYS}
        self.state_dddot = {key: 0.0 for key in JERK_KEYS}
        self.obs, self.obs_dict = self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1)
        # 1. State, disturbance
        x = np.array([self.state[key] for key in STATE_KEYS], dtype=np.float32)
        z = self.calculate_disturbance(self.state.copy())

        # 2. Calculate Human Control input
        u_human = self.human_controller(self.obs_dict, self.state)

        # 3. Bump detection
        if not self.bump_detected:
            self.detect_bump(x, u_human, z)
            if self.bump_detected:
                self.eride_time = 0.0

        # 4. Calculate total control input
        u_eride = 0.0
        if self.bump_detected:
            # s = np.clip((self.eride_time - (self.eride_duration - self.eride_decay)) / self.eride_decay, 0.0, 1.0)
            # decay = 1.0 - (3.0*s**2 - 2.0*s**3)
            # u_eride = action * decay
            # Erase Decay (이게 아무래도 강화학습 입장에서 학습하기 어려움 것 같음)
            u_eride = float(action[0])
        u = u_human + u_eride

        # 5. Vehicle dynamics
        dx = np.asarray(self.vehicle(x, u, z), dtype=np.float32)
        x_next = x + dx * self.dt

        # 6. Update
        for i, key in enumerate(STATE_KEYS):
            self.state[key] = x_next[i]

        prev_ddot = self.state_ddot.copy()
        for i, key in enumerate(ACCEL_KEYS):
            self.state_ddot[key] = dx[i]
        for i, key in enumerate(JERK_KEYS):
            accel_key = ACCEL_KEYS[i]
            self.state_dddot[key] = (self.state_ddot[accel_key] - prev_ddot[accel_key]) / self.dt

        self.obs, self.obs_dict = self._get_obs()

        # 7. Calculate reward
        reward = self._get_reward()

        # 8. Update time
        if self.bump_detected:
            self.eride_time += self.dt
            if self.eride_time >= self.eride_duration:
                self.eride_time = 0.0
                self.bump_detected = False

        self.time += self.dt

        truncated = True if self.time >= self.max_time else False
        if truncated:
            print(f"time: {self.time:.2f}")
            print(f"eride time: {self.eride_time:.2f}")
            print(f"Bump detected: {self.bump_detected}")
        info = self._get_info(z, reward, u_eride, u_human)
        return self.obs.copy(), reward, False, truncated, info

    def _get_obs(self):
        obs_values = []
        obs_dict = {}
        for key in self.obs_keys:
            if key in self.state:
                value = self.state[key]
            elif key in self.state_ddot:
                value = self.state_ddot[key]
            elif key in self.state_dddot:
                value = self.state_dddot[key]
            else:
                raise KeyError(f"Key '{key}' not found in state, state_ddot, or state_dddot.")
            obs_values.append(value)
            obs_dict[key] = value
        obs_array = np.array(obs_values, dtype=np.float32)
        return obs_array, obs_dict

    def _get_info(self, z, reward, u_eride, u_human):
        return {
            "time": self.time, "bump_detected": self.bump_detected,
            "state": self.state.copy(), "state_ddot": self.state_ddot.copy(), "state_dddot": self.state_dddot.copy(),
            "disturbance": z, "reward": reward, "u_eride": u_eride, "u_human": u_human
        }

    def _get_reward(self):
        pitch_rate = self.state["dtheta"]
        pitch_penalty = - (pitch_rate * 5) ** 2
        return pitch_penalty

    def detect_bump(self, x, u, z):
        x_pred = self.vehicle(x, u, z=[0, 0]).copy()
        x_real = self.vehicle(x, u, z=z)
        residual1 = np.abs(x_pred[2] - x_real[2])  # dz_us_f
        residual2 = np.abs(x_pred[3] - x_real[3])  # dz_us_r
        self.bump_detected = residual1 > 0.05 or residual2 > 0.05

    def calculate_disturbance(self, state):
        x_front = state["x_com"] + self.vehicle_params.l_f * np.cos(state["theta"])
        x_rear = state["x_com"] - self.vehicle_params.l_r * np.cos(state["theta"])
        return np.array([self.bump(x_front), self.bump(x_rear)], dtype=np.float32)

class SingleScenarioEnv(SuspensionEnv):
    def __init__(self, human_controller=None):
        super().__init__(human_controller=human_controller, is_multi_bump=False)

        self.t_observe    = self.config.t_observe
        self.observe_step = int(self.t_observe / self.config.dt_inner)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        while not self.bump_detected:
            obs, reward, terminated, truncated, info = super().step(0.0)
            if terminated or truncated:
                raise RuntimeError("Bump not detected within the initial steps. Check the bump configuration.")
        self.current_step = 0
        return obs, info
    
    def step(self, u_eride):
        obs, reward, terminated, truncated, info = super().step(u_eride)
        self.current_step += 1
        if self.current_step >= self.observe_step:
            truncated = True

        return obs, reward, terminated, truncated, info