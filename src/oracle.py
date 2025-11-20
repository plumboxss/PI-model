import yaml
import numpy as np

class Oracle:
    """
    YAML 설정 파일에 정의된 가중치를 기반으로 궤적에 대한 보상을 계산하는 클래스.
    """
    def __init__(self, weights):
        """
        Oracle을 초기화합니다.

        Args:
            weights (dict): 각 피처에 대한 가중치를 담은 딕셔너리.
        """
        self.weights = weights

    def __call__(self, result_for_oracle):
        """
        주어진 결과에 대해 보상, 확률, 응답을 계산합니다.
        현재 구현에서는 특징 기반 보상을 계산하는 데 초점을 맞춥니다.

        Args:
            result_for_oracle (dict): 'features' 키를 포함해야 하는 딕셔너리.

        Returns:
            tuple: (보상, 확률, 응답). 확률과 응답은 현재 플레이스홀더입니다.
        """
        features = result_for_oracle.get("features", {})
        if not features:
            return 0.0, 0.5, "No features provided"

        reward = 0.0
        for feature, weight in self.weights.items():
            if feature in features:
                # 가중치가 음수이면 해당 피처 값이 작을수록 보상이 높아짐
                # 가중치가 양수이면 해당 피처 값이 클수록 보상이 높아짐
                reward += features[feature] * weight
        
        # 확률과 응답은 현재 사용되지 않으므로 더미 값을 반환
        probability = 0.5 
        response = "OK"

        return reward, probability, response

def create_oracle_from_config(config_name):
    """
    YAML 설정 파일 이름으로 Oracle 객체를 생성합니다.
    파일은 프로젝트 루트에 있다고 가정합니다.

    Args:
        config_name (str): 'A', 'B' 등 Oracle의 이름. `oracle_A.yaml` 형식으로 파일을 찾습니다.

    Returns:
        Oracle: 생성된 Oracle 객체.
    """
    config_path = f"oracle_{config_name}.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            weights = config.get('weights', {})
            return Oracle(weights)
    except FileNotFoundError:
        raise FileNotFoundError(f"Oracle config file not found at {config_path}")
    except Exception as e:
        raise IOError(f"Error reading or parsing oracle config file: {e}")
