import gymnasium
import numpy as np
import torch

def _get_obs_from_env(env, key: str):
    if key in env.state:       return env.state[key]
    if key in env.state_ddot:  return env.state_ddot[key]
    if key in env.state_dddot: return env.state_dddot[key]
    raise KeyError(f"Key '{key}' not found in state/state_ddot/state_dddot")

class KeyedObservationWrapper(gymnasium.Wrapper):
    def __init__(self, env, keys, history_length=1, mean=None, std=None):
        super().__init__(env)
        self.base_env = self.env.unwrapped
        self.keys = list(keys)
        self.mean = np.array(mean, dtype=np.float32) if mean is not None else None
        self.std  = np.array(std,  dtype=np.float32) if std  is not None else None
        self.history_length = history_length
        self.history = []
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.keys) * history_length,), dtype=np.float32
        )

    def _obs(self):
        current_obs = np.array([_get_obs_from_env(self.base_env, k) for k in self.keys], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            current_obs = (current_obs - self.mean) / (self.std + 1e-8)
        
        self.history.append(current_obs)
        if len(self.history) > self.history_length:
            self.history.pop(0)
        
        while len(self.history) < self.history_length:
            self.history.insert(0, current_obs)
        
        return np.concatenate(self.history)

    def reset(self, **kwargs):
        self.history = []
        _, info = self.env.reset(**kwargs)
        return self._obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._obs(), reward, terminated, truncated, info

class ControllerWrapper(gymnasium.Wrapper):
    """
    mode="schedule": RL action -> controller.schedule(action) -> controller로 토크 산출 -> 플랜트에 전달
    mode="pure":     RL action 없이 controller 만으로 토크 산출(정책 무시, 스크립트 컨트롤러 평가용)
    """
    def __init__(self, env, controller, mode: str = "schedule"):
        super().__init__(env)
        assert mode in ("schedule", "pure")
        self.base_env = self.env.unwrapped
        self.controller = controller
        self.mode = mode
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) if mode=="schedule" \
                            else gymnasium.spaces.Box(low=0.0, high=0.0, shape=(0,), dtype=np.float32)

    def reset(self, **kwargs):
        self.controller.reset()
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.mode == "schedule":
            self.controller.schedule(action)
        u = self.controller(self.base_env.obs_dict, self.base_env.state)

        obs, r, term, trunc, info = self.env.step(float(u))
        info = dict(info)

        info.update({"controller": self.controller.name, "u_eride": float(u)})
        info.update({f"ctrl_{k}": v for k, v in self.controller.get_params().items()})
        return obs, r, term, trunc, info

class LearnedRewardWrapper(gymnasium.Wrapper):
    def __init__(self, env, reward_model, mix=None):
        super().__init__(env)
        self.base_env = self.env.unwrapped
        self.reward_model = reward_model
        self.mix = mix

    def reset(self, **kw):
        return self.env.reset(**kw)

    def step(self, action):
        obs, r_env, term, trunc, info = self.env.step(action)
        x = self._get_reward_input(info)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            r_model = self.reward_model(x_tensor).detach().cpu().numpy().item()
        r = r_model if self.mix is None else self.mix * float(r_env) + (1.0 - self.mix) * float(r_model)
        info = dict(info)
        info["reward_env"] = float(r_env)
        info["reward_model"] = float(r_model)
        info["reward_mixed"] = float(r)
        return obs, r, term, trunc, info
    
    def _get_reward_input(self, info):
        state = info['state']
        ddots = info['state_ddot']
        x = np.array([state['dtheta'], ddots['ddx_com'], ddots['ddz_com']])
        return x