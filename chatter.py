import os

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
        return res

    def _prep_for_ft(self, text):
        prepared_text = []
        for token in NLPOps.nlp(text):
            if token.pos_ in ["VERB", "ADJ", "NOUN"]:
                prepared_text.append(token.lemma_)
        return prepared_text

    def fulltext_search(self, query, n):
        top_docs = self.get_ft().get_top_n(documents=self.snippets, query=self._prep_for_ft(query), n=n)
        return top_docs

    def get_answer(self, sys_prompt, user_prompt):
        r = self.cli.chat.completions.create(
                model=self.llm,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}])
        return r.choices[0].message.content

    def search_recipe(self, query, n_of_ft_snippets, n_of_ss_snippets):
        ss = self.semantic_search(query, n_of_ss_snippets)
        ft = self.fulltext_search(query, n_of_ft_snippets)
        ss_titles = []
        for i in ss[0]:
            ss_titles.append(self.snippets[i["corpus_id"]][0])

        print(f"######################################### SEMANTIC SEARCH")
        for tit in ss_titles:
            for name, sets in self.settings.items():
                if tit == name:
                    print(tit)
                    print(sets)
                    print("--------")
        print("########################################### FULLTEXT SEARCH")
        for i in ft:
            for name, sets in self.settings.items():
                if i[0] == name:
                    print(i[0])
                    print(sets)
                    print("-------")

ctr = Chatter(oai_model="gpt-4o-mini",
              embedding_model_dir="multilingual-e5-base",
              embeddings=DataOps.load_npy(os.path.join("data", "embeddings.npy")),
              snippets=DataOps.load_json(os.path.join("data", "snippets.json")),
              settings=DataOps.load_json(os.path.join("data", "settings.json")),
              simulations=DataOps.load_json(os.path.join("data", "simulations.json")))

# ctr.semantic_search("I am looking for colder colors and vintage look. I will be taking pictures of nature, forests and lakes")
# ctr.fulltext_search("I am looking for colder colors and vintage look. I will be taking pictures of nature, forests and lakes")
ctr.search_recipe("I am looking for colder colors and vintage look. I will be taking pictures of nature, forests and lakes", 3, 3)


