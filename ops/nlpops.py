import spacy
from .dataops import load_yaml


def load_nlp(spacy_model):
    try:
        return spacy.load(spacy_model)
    except OSError as e:
        raise RuntimeError(f"Issue with loading NLP. Model seems not to be installed. Failed with {e}")

prompts = load_yaml("prompts.yaml")
