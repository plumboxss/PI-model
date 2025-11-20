import numpy as np

class PController:
    """
    간단한 비례 제어기(Proportional Controller) 클래스.
    """
    def __init__(self, kp):
        """
        제어기를 초기화합니다.

        Args:
            kp (float): 비례 게인(P gain).
        """
        self.kp = kp

    def control(self, state):
        """
        현재 상태를 기반으로 제어 입력을 계산합니다.
        여기서는 단순화를 위해 상태의 첫 번째 요소를 오차라고 가정합니다.
        실제 차량 모델에 맞게 이 로직을 조정해야 할 수 있습니다.

        Args:
            state (np.array): 현재 시스템의 상태 벡터.

        Returns:
            np.array: 계산된 제어 입력 (액션).
        """
        # 이 예제에서는 상태 벡터의 첫 번째 값을 사용하여 제어 신호를 생성합니다.
        # 실제 구현에서는 특정 상태 변수(예: 피치 각도)를 사용해야 합니다.
        error = state[6] # 'theta' (pitch)를 오차로 가정
        action = -self.kp * error
        return np.array([action])
