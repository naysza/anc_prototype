import yaml


def load_config(path: str) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration represented as a Python dictionary.
    """

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration file must contain a YAML dictionary."
        )

    return config
