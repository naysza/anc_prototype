import argparse
import sys
from anc.utils.io import load_config_yaml
from anc.experiments.runner import DictConfiguration, DummyANCExperiment

def main():
    parser = argparse.ArgumentParser(description="ANC Prototype Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    try:
        config_dict = load_config_yaml(args.config)
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    config = DictConfiguration(config_dict)
    
    experiment_name = config.get("experiment_name", "unknown")
    print(f"Starting experiment: {experiment_name}")
    
    # In a real system, we'd have a factory for experiments based on config.
    experiment = DummyANCExperiment()
    result = experiment.run(config)
    
    print("Experiment completed successfully.")

if __name__ == "__main__":
    main()
