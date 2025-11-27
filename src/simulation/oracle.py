import yaml
import numpy as np

class Oracle:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, result_for_oracle):
        features = result_for_oracle.get("features", {})
        if not features:
            return 0.0, 0.5, "No features provided"
        
        reward = 0.0
        for feature, weight in self.weights.items():
            if feature in features:
                reward += features[feature] * weight
                
        probability = 0.5 # Placeholder
        response = "OK"
        return reward, probability, response

def create_oracle_from_config(config_name):
    config_path = f"configs/oracle_{config_name}.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            weights = config.get('weights', {})
            return Oracle(weights)
    except FileNotFoundError:
        raise FileNotFoundError(f"Oracle config file not found at {config_path}")
    except Exception as e:
        raise IOError(f"Error reading or parsing oracle config file: {e}")

