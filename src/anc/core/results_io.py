import json
import os
from typing import Dict, Any

def save_result_json(result_dict: Dict[str, Any], filepath: str) -> None:
    """Saves a dictionary of results to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=4)
        
def load_config_yaml(filepath: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    import yaml
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)
