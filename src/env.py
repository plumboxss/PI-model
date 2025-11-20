import gym
from gym import spaces
import numpy as np

# 프로젝트 루트에 있는 plant.py를 import하기 위해 경로 추가
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plant import ODESystem, create_plant_from_config

class SingleScenarioEnv(gym.Env):
    """
    단일 시나리오 차량 시뮬레이션을 위한 Gym 환경 래퍼.
    """
    def __init__(self):
        super(SingleScenarioEnv, self).__init__()

        # plant.py에서 실제 차량 모델 시스템을 로드
        self.plant = create_plant_from_config()
        self.ode_system = ODESystem(self.plant)

        # 상태 및 행동 공간 정의 (plant.py와 일치해야 함)
        # 상태 공간의 차원은 plant의 초기 상태 벡터 크기와 동일
        state_dim = len(self.plant.get_initial_state())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        # 행동 공간은 단일 스칼라 값(제어 토크)이라고 가정
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.time_step = self.plant.time_step
        self.current_time = 0.0
        self.state = None
        self.reset()

    def reset(self):
        """
        환경을 초기 상태로 리셋합니다.
        """
        self.state = self.plant.get_initial_state()
        self.current_time = 0.0
        return self.state

    def step(self, action):
        """
        환경에서 한 타임스텝을 진행합니다.

        Args:
            action (np.array): 에이전트가 취한 행동.

        Returns:
            tuple: (다음 상태, 보상, 종료 여부, 추가 정보).
        """
        # ODE 시스템을 사용하여 다음 상태 계산
        t_span = [self.current_time, self.current_time + self.time_step]
        
        # plant.py의 `ode_func`는 `t`, `y`, `u`를 인자로 받음
        # 여기서 `y`는 현재 상태, `u`는 행동(제어 입력)
        next_state, info = self.ode_system.step(self.state, action, t_span)
        
        self.state = next_state
        self.current_time += self.time_step

        # 보상은 현재 이 환경 수준에서 정의되지 않음 (상위 레벨에서 계산)
        reward = 0.0

        # 종료 조건 (예: 시뮬레이션 시간 초과)
        done = self.current_time >= self.plant.T

        # info 딕셔너리에 plant에서 반환된 추가 정보 포함
        return self.state, reward, done, info

    def render(self, mode='human'):
        # 렌더링은 현재 구현되지 않음
        pass

    def close(self):
        pass
