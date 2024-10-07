import os
import re

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

from ops import DataOps, NLPOps


class Chatter:
    def __init__(self, oai_model: str, embedding_model_dir, embeddings, snippets, settings, simulations):
        self.llm = oai_model
        self.cli = OpenAI()
        self.model = SentenceTransformer(embedding_model_dir)
        self.ft_model = None
        self.embeddings = embeddings
        self.snippets = snippets
        self.settings = settings
        self.sims = simulations

    def get_ft(self):
        if not self.ft_model:
            corpus = [s[1].split(", ") for s in self.snippets]
            self.ft_model = BM25Okapi(corpus)
        return self.ft_model

    def semantic_search(self, query, n):
        embed_query = self.model.encode(f"query: {query}", convert_to_tensor=True)
        res = util.semantic_search(embed_query, self.embeddings, top_k=n)
        return res[0]

    def _prep_for_ft(self, text):
        prepared_text = []
        for token in NLPOps.nlp(text):
            if token.pos_ in ["VERB", "ADJ", "NOUN"]:
                prepared_text.append(token.lemma_)
        return prepared_text

    def fulltext_search(self, query, n):
        scores = self.get_ft().get_scores(self._prep_for_ft(query))
        top_n = np.argsort(scores)[::-1]
        res = []
        known_tits = []
        for i in top_n:
            current_title = self.snippets[i][0]
            if current_title not in known_tits:
                res.append({"corpus_id": i, "score": scores[i]})
                known_tits.append(current_title)
            if len(res) == n:
                break
        return res

    def get_answer(self, sys_prompt, user_prompt):
        r = self.cli.chat.completions.create(
                model=self.llm,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}])
        return r.choices[0].message.content

    def reg_matches(self, cam_type, sim_tit):
        return bool(re.match(fr".*{cam_type}.*", sim_tit))

    def search_recipe(self, query, n_of_ft_snippets, n_of_ss_snippets, camera: list):
        ss = self.semantic_search(query, n_of_ss_snippets)
        ft = self.fulltext_search(query, n_of_ft_snippets)
        titles = []
        for meta in ss + ft:
            current_title = self.snippets[meta["corpus_id"]][0]
            if any(self.reg_matches(c, current_title) for c in camera):
                titles.append(current_title)

        for tit in titles:
            for name, sets in self.settings.items():
                if tit == name:
                    print(tit)
                    print(sets)
                    print("--------")

ctr = Chatter(oai_model="gpt-4o-mini",
              embedding_model_dir="multilingual-e5-base",
              embeddings=DataOps.load_npy(os.path.join("data", "embeddings.npy")),
              snippets=DataOps.load_json(os.path.join("data", "snippets.json")),
              settings=DataOps.load_json(os.path.join("data", "settings.json")),
              simulations=DataOps.load_json(os.path.join("data", "simulations.json")))

# ctr.semantic_search("I am looking for colder colors and vintage look. I will be taking pictures of nature, forests and lakes")
# ctr.fulltext_search("I am looking for colder colors and vintage look. I will be taking pictures of nature, forests and lakes")
ctr.search_recipe("Vivid and vibrant colours. A bit of constrast, versatile usage. X-T30 or X-Trans IV",
                  10,
                  10,
                  ["X-T1", "X-Trans II"])

# TODO: same found indices
# TODO: camera/sensor type -- DONE
# TODO: implemet retrieval threshold
