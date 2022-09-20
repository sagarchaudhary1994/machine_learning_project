import yaml
from housing.exception import Housing_Exception
import os
import sys


def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file and returns the content as dictionary.
    file_path: str
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise Housing_Exception(e, sys) from e
