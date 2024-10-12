import json
import numpy as np
import argparse
import spacy

import yaml


def export_json(data, filename):
    with open(filename, "w+", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def export_npy(data, filename):
    with open(filename, "wb+") as f:
        np.save(f, data)


def load_npy(filename):
    with open(filename, "rb") as f:
        return np.load(filename)


def parse_my_args(arg_template):
    """
    Parse script args.
    :param arg_template: List of lists of arg info: [[arg_name: str, arg_type: type, arg_required: bool], [...], ...]
    :return: Dict of parsed args
    """
    parser = argparse.ArgumentParser()
    for arg_name, arg_type, arg_required in arg_template:
        parser.add_argument(arg_name, type=arg_type, required=arg_required)
    return parser.parse_args()


def load_yaml(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def load_nlp(spacy_model):
    try:
        return spacy.load(spacy_model)
    except OSError as e:
        raise Exception(f"Unable to load spaCy model:\n{e}\nTo download: https://spacy.io/usage/models")


prompts = load_yaml("prompts.yaml")
