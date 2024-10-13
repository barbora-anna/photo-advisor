import os
import re

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from pprint import pprint

import ops


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
        self.nlp = ops.load_nlp("en_core_web_sm")

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
        for token in self.nlp(text):
            if token.pos_ in ["VERB", "ADJ", "NOUN"]:
                prepared_text.append(token.lemma_)
        return prepared_text

    def fulltext_search(self, query):
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
        return res

    def get_answer(self, sys_prompt, user_prompt):
        r = self.cli.chat.completions.create(
                model=self.llm,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}])
        return r.choices[0].message.content

    def _reg_matches(self, cam_type, sim_tit):
        return bool(re.match(fr".*{cam_type}.*", sim_tit))

    def _get_n_of_semantic_chunks(self, ft_res, default_ss_chunks=10):
        if not ft_res:
            return default_ss_chunks
        return len(ft_res)

    def search_recipe(self, query, camera: list):
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
        return res

    def _format_settings_for_prompt(self, recipes_res):
        settings = []
        for r in recipes_res:
            settings.append(r.get("settings"))
        return "\n*************\n".join(settings)


    def get_recommendation(self, query, camera: list):
        recipes = self.search_recipe(query, camera)
        user_prompt = f"USER DESCRIPTION: {query} /// RECIPE EXAMPLES: {self._format_settings_for_prompt(recipes)}"
        msgs = [
            {"role": "system", "content": ops.prompts["recipe_creation"]["sys"]},
            {"role": "user", "content": user_prompt}]

        r = self.cli.chat.completions.create(model="gpt-4o-mini", messages=msgs)
        return r.choices[0].message.content


if __name__ == "__main__":
    ctr = Chatter(oai_model="gpt-4o-mini",
                  embedding_model_dir="multilingual-e5-base",
                  embeddings=ops.load_npy(os.path.join("data", "embeddings.npy")),
                  snippets=ops.load_json(os.path.join("data", "snippets.json")),
                  settings=ops.load_json(os.path.join("data", "settings.json")),
                  simulations=ops.load_json(os.path.join("data", "simulations.json")))

    # ctr.semantic_search("I am looking for colder colors and vintage look. I will be taking pictures of nature, forests and lakes")
    # ctr.fulltext_search("I am looking for colder colors and vintage look. I will be taking pictures of nature, forests and lakes")
    # ctr.search_recipe("Autumn colours. Warm and vivid with a bit of contrast. Pictures of nature, mushrooms, trees.",
    #                   ["X-T30", "X-Trans IV"])
    # ctr.search_recipe("I would like to take pictures of architecture, streets of cities. I sought for vivid colors and bright lights.",
    #                   ["X-T1", "X-Trans II"])

    print(ctr.get_recommendation("I need something for misty mornings. I like when white is white without tint. I also want vibrant and lively colours.", ["X-T30", "X-Trans IV"]))

    # while True:
    #     inp = input("What are you looking for?")
    #     ctr.search_recipe(inp, camera=["X-T30", "X-Trans IV"])
