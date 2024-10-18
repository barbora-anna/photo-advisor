import os
import re
import sys

import logging
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from pprint import pprint

from ops import dataops as dto
from ops import nlpops as nlo

log = logging.getLogger("Chatter")
logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s')


class Chatter:
    def __init__(self, embedding_model: str, embeddings: np.array, snippets: list[list[str]],
                 settings: dict[str, str], simulations: list[dict[str, str]], default_semantic_chunks: int = 10,
                 oai_model: str = "gpt-4o-mini"):
        """

        :param embedding_model: Dirname of local embedding model / huggingface identifier
        :param embeddings: Embedded
        :param snippets:
        :param settings:
        :param simulations:
        :param default_semantic_chunks:
        :param oai_model:
        """
        try:
            self.model = SentenceTransformer(embedding_model)
            self.nlp = nlo.load_nlp("en_core_web_smm")
        except Exception:
            # todo: raise my exception
            log.exception(f"Unable to load model.")
            sys.exit(1)

        if not os.getenv("OPENAI_API_KEY"):
            log.error("Provide OpenAI API key in env!")
            sys.exit(1)

        log.info("Loaded embedding and NLP models..")
        self.ft_model = None
        self.embeddings = embeddings
        self.snippets = snippets
        self.settings = settings
        self.sims = simulations
        self.llm = oai_model
        self.cli = OpenAI()
        self.default_ss_chunks = default_semantic_chunks

    def get_ft(self) -> BM25Okapi:
        """Return fulltext search model."""
        if not self.ft_model:
            corpus = [s[1].split(", ") for s in self.snippets]
            self.ft_model = BM25Okapi(corpus)
            log.info("Loaded fulltext model...")
        return self.ft_model

    def semantic_search(self, query: str, n: int) -> list[dict[str, int | float]]:
        """Run semantic search. Return metadata: index, confidence"""
        embed_query = self.model.encode(f"query: {query}", convert_to_tensor=True)
        res = util.semantic_search(embed_query, self.embeddings, top_k=n)
        log.debug("Finished semantic search...")
        return res[0]

    def _prep_for_ft(self, text: str) -> list[str]:
        """Prepare text for fulltext search. Find valid POS, return the lemmas."""
        prepared_text = []
        for token in self.nlp(text):
            if token.pos_ in ["VERB", "ADJ", "NOUN"]:
                prepared_text.append(token.lemma_)
        return prepared_text

    def fulltext_search(self, query: str) -> list[dict[str, int | float]]:
        """
        Run fulltext search. Return metadata: index, confidence.
        Reformat algorithm's result for semantic search compatibility.
        """
        scores = self.get_ft().get_scores(self._prep_for_ft(query))
        top_n = np.argsort(scores)[::-1]
        res = []
        known_tits = []
        for i in top_n:
            if scores[i] < 1:
                break
            current_title = self.snippets[i][0]
            if current_title not in known_tits:
                res.append({"corpus_id": i, "score": scores[i]})
                known_tits.append(current_title)
        log.debug("Finished fulltext search...")
        return res

    def get_answer(self, sys_prompt: str, user_prompt: str) -> str:
        """Get LLM answer from OpenAI API."""
        r = self.cli.chat.completions.create(
                model=self.llm,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}])
        log.debug("Got OAI answer...")
        return r.choices[0].message.content

    def _reg_matches(self, cam_type: str, sim_tit: str) -> bool:
        return bool(re.match(fr"\s{cam_type}\s", sim_tit))

    def _get_n_of_semantic_chunks(self, ft_res: list[dict[str, int | float]]) -> int:
        if not ft_res:
            return self.default_ss_chunks
        return len(ft_res)

    def search_recipe(self, query: str, camera: list) -> list[dict[str, str]]:
        ft = self.fulltext_search(query)
        ss = self.semantic_search(query, self._get_n_of_semantic_chunks(ft))
        titles = []
        for meta in ss + ft:
            current_title = self.snippets[meta["corpus_id"]][0]
            if any(self._reg_matches(c, current_title) for c in camera):
                if current_title not in titles:
                    titles.append(current_title)
        res = []
        for tit in titles:
            for name, sets in self.settings.items():
                if tit == name:
                    res.append({
                        "title": tit,
                        "settings": sets})
        log.debug(f"Got {len(res)} valid recipes...")
        return res

    def _format_settings_for_prompt(self, recipes_res: list[dict[str, str]]) -> str:
        settings = []
        for r in recipes_res:
            settings.append(r.get("settings"))
        return "\n*************\n".join(settings)

    def get_recommendation(self, query: str, camera: list) -> str | None:
        recipes = self.search_recipe(query, camera)
        user_prompt = f"USER DESCRIPTION: {query} /// RECIPE EXAMPLES: {self._format_settings_for_prompt(recipes)}"
        msgs = [
            {"role": "system", "content": nlo.prompts["recipe_creation"]["sys"]},
            {"role": "user", "content": user_prompt}]

        r = self.cli.chat.completions.create(model="gpt-4o-mini", messages=msgs)
        log.debug("Received OAI answer...")
        return r.choices[0].message.content


if __name__ == "__main__":
    args = dto.parse_my_args([
        ["--debug", str, False],
        ["--to-do", str, False]])

    if args.debug:
        log.setLevel("DEBUG")

    ctr = Chatter(oai_model="gpt-4o-mini",
                  embedding_model_dir="multilingual-e5-base",
                  embeddings=ops.load_npy(os.path.join("data", "embeddings.npy")),
                  snippets=ops.load_json(os.path.join("data", "snippets.json")),
                  settings=ops.load_json(os.path.join("data", "settings.json")),
                  simulations=ops.load_json(os.path.join("data", "simulations.json")))

    print(ctr.get_recommendation("I need something for misty mornings. I like when white is white without tint. I also want vibrant and lively colours.", ["X-T30", "X-Trans IV"]))
