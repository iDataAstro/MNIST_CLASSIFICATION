import yaml
import os
import logging
import time
import json


def read_config(config_path: str) -> dict:
    """
    Args:
        config_path: Path to the config file
    Returns:
        content: Returns content of config file
    """
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)

    logging.info(f"yaml file: {config_path} loaded successfully")
    return content


def get_unique_name(stage: str) -> str:
    """
    Returns:
        Returns unique name with current time stamp
    """
    return time.strftime(stage + '_%Y%m%d_%H%M%S')


def create_directories(path_to_directories: list) -> None:
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logging.info(f"created directory at: {path}")


def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")
