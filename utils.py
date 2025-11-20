import argparse
import numpy as np
from tqdm import tqdm

class SimulationRecorder:
    """
    시뮬레이션 전체 과정을 실행하고 모든 데이터를 기록하는 클래스.
    """
    def __init__(self, env, controller, downsample=1):
        self.env = env
        self.controller = controller
        self.downsample = downsample
        self.results = {
            'time': [],
            'state_all': [],
            'state_ddot_all': [],
            'state_dddot_all': [],
            'action_all': [],
            'reward_all': [],
        }

    def simulate(self, seed=None):
        if seed is not None:
            # env.seed(seed) is not a standard gym method, but can be added if needed
            np.random.seed(seed)
            
        obs = self.env.reset()
        done = False
        
        # tqdm을 사용하여 시뮬레이션 진행 상황 표시
        num_steps = int(self.env.plant.T / self.env.time_step)
        with tqdm(total=num_steps, desc="Simulating Episode", leave=False) as pbar:
            while not done:
                action = self.controller.control(obs)
                obs, reward, done, info = self.env.step(action)
                
                # 모든 타임스텝의 데이터를 기록
                self.results['time'].append(self.env.current_time)
                self.results['state_all'].append(obs)
                self.results['action_all'].append(action)
                self.results['reward_all'].append(reward)
                
                # info 딕셔너리에서 추가적인 미분 값들을 가져옴
                self.results['state_ddot_all'].append(info.get('state_ddot', np.zeros_like(obs)))
                self.results['state_dddot_all'].append(info.get('state_dddot', np.zeros_like(obs)))

                pbar.update(1)

        # 결과를 numpy 배열로 변환
        for key, val in self.results.items():
            self.results[key] = np.array(val)

        # 다운샘플링 (필요한 경우)
        if self.downsample > 1:
            self.results['state'] = self.results['state_all'][::self.downsample]
            self.results['state_ddot'] = self.results['state_ddot_all'][::self.downsample]
        else:
            self.results['state'] = self.results['state_all']
            self.results['state_ddot'] = self.results['state_ddot_all']

    def __getitem__(self, key):
        return self.results[key]

def generate_data_parser():
    """
    generate_data.py 스크립트를 위한 ArgumentParser를 생성합니다.
    """
    parser = argparse.ArgumentParser(description="Generate trajectory data using vehicle simulation.")
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of simulation episodes to run.')
    parser.add_argument('--oracle-name', type=str, required=True, help="Name of the oracle to use (e.g., 'A', 'B'). Corresponds to 'oracle_A.yaml'.")
    parser.add_argument('--dataset-name', type=str, default=None, help='Name for the generated dataset file.')
    return parser

def visualize_oracle_data(dataset_path, save_dir):
    """
    생성된 데이터셋을 시각화하는 함수 (플레이스홀더).
    """
    print(f"[Placeholder] Visualizing data from {dataset_path} and saving to {save_dir}...")
    # 여기에 matplotlib 등을 사용하여 데이터를 플로팅하는 코드를 추가할 수 있습니다.
    pass
