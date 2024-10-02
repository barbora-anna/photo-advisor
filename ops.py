import json
import numpy as np
import argparse
import spacy

import yaml


class DataOps:
    @staticmethod
    def export_json(data, filename):
        with open(filename, "w+", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load_json(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def export_npy(data, filename):
        with open(filename, "wb+") as f:
            np.save(f, data)

    @staticmethod
    def load_npy(filename):
        with open(filename, "rb") as f:
            return np.load(filename)

    @staticmethod
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

    @staticmethod
    def load_yaml(filename):
        with open(filename, "r") as f:
            return yaml.safe_load(f)


class NLPOps:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as e:
        raise Exception(f"Unable to load spaCy model:\n{e}\nTo download: https://spacy.io/usage/models")

    prompts = DataOps.load_yaml("prompts.yaml")
