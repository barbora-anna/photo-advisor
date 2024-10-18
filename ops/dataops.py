import json
import os

import chromadb
from chromadb.utils import embedding_functions
import numpy as np
import argparse

import yaml


def export_json(data: list | dict, filename: str | os.PathLike) -> None:
    with open(filename, "w+", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(filename: str | os.PathLike) -> dict | list:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def export_npy(data, filename: str | os.PathLike) -> None:
    with open(filename, "wb+") as f:
        np.save(f, data)


def load_npy(filename) -> None:
    with open(filename, "rb") as f:
        return np.load(f)


def parse_my_args(arg_template: list[list[str | type | bool]]):
    """
    Parse script args.
    :param arg_template: List of lists of arg info: [[arg_name, arg_type, arg_required], [...], ...]
    :return: Dict of parsed args
    """
    parser = argparse.ArgumentParser()
    for arg_name, arg_type, arg_required in arg_template:
        parser.add_argument(arg_name, type=arg_type, required=arg_required)
    return parser.parse_args()


def load_yaml(filename) -> list | dict:
    with open(filename, "r") as f:
        return yaml.safe_load(f)
