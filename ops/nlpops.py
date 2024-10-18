import spacy
from .dataops import load_yaml


def load_nlp(spacy_model):
    return spacy.load(spacy_model)


prompts = load_yaml("prompts.yaml")
