import gym
from gym import spaces
import numpy as np

from src.simulation.plant import create_plant_from_config

class SingleScenarioEnv(gym.Env):
    def __init__(self):
        super(SingleScenarioEnv, self).__init__()
        self.plant = create_plant_from_config()
        
        # State dim: 10
        # Plant state initialized as zeros(10) in plant.py
        state_dim = 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        # Action dim: 1 (u)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Get time_step from params if available, else default to 0.01
        self.time_step = getattr(self.plant.params, 'time_step', 0.01)
        self.current_time = 0.0
        self.state = None
        
        self.reset()

    def reset(self):
        self.state = self.plant.reset()
        self.current_time = 0.0
        return self.state

    def step(self, action):
        # action is numpy array or scalar
        if isinstance(action, np.ndarray):
            u = action.item()
        else:
            u = action
            
        self.state = self.plant.step(u, self.time_step)
        self.current_time += self.time_step
        
        # T from params, default to 10.0 if not set
        max_time = getattr(self.plant.params, 'T', 10.0)
        done = self.current_time >= max_time
        
        reward = 0.0 # Reward logic is separate
        
        return self.state, reward, done, {}
