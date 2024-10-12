import logging
import os
from http.client import HTTPConnection

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import ops

log = logging.getLogger("FujiXWeeklyScraper")
HTTPConnection.debuglevel = 0
logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s')


class FujiXWeeklyScraper:
    def __init__(self, snippets=None, keyword_llm_eval="gpt-4o-mini"):
        """
        Scrape the website and prepare data for search.
        :param snippets: Scraped snippets can be passed if scraping has done already and just modifications are needed.
        :param keyword_llm_eval: This model will create keywords for fulltext search
        """
        self.url_ = "https://fujixweekly.com/sitemap-1.xml"
        self.snippets = snippets if snippets else []
        self.stopwords = ["Share this", "Nobody pays", "Help Fuji X", "Click to share", "*Related*", "!["]
        self.oai_cli = OpenAI()
        self.kwd_llm = keyword_llm_eval
        self.nlp = ops.load_nlp("en_core_web_sm")

    def get_recipe_urls(self):
        recipe_urls = []
        for link in [i.text for i in self.get_soup(self.url_, "xml").findAll("loc")]:
            if "simulation-recipe-" in link:
                recipe_urls.append(link)
        log.debug(f"Got {len(recipe_urls)} recipe urls")
        return recipe_urls

    def get_page(self, url_):
        r = requests.get(url_)
        if not r.ok:
            raise RuntimeError(f"Couldn't retrieve page {url_}")
        return r

    def get_soup(self, url, parser="html.parser"):
        res = self.get_page(url)
        return BeautifulSoup(res.content, parser)

    def _get_sim_title(self, sim_soup):
        return str(sim_soup.find("h1", {"class": "entry-title"}).text)

    def _get_sim_description(self, sim_soup):
        description = []
        for i in sim_soup.find("div", {"class": "entry-content"}):
            if i.text:
                if self.contains_verb(i.text) and not any([w in md(str(i)) for w in self.stopwords]):
                    description.append(md(str(i)))
        return " ".join(description)

    def _get_sim_settings(self, sim_soup):
        settings = [md(str(i)) for i in sim_soup.find("div", {"class": "entry-content"}).findAll("strong")]
        return "\n".join(settings)

    def contains_verb(self, text):
        for token in self.nlp(text):
            if token.pos_ == "VERB":
                return True
        return False

    def get_snippets(self):
        if not self.snippets:
            counter = 0
            for url in self.get_recipe_urls():
                soup = self.get_soup(url)
                sim_desc = self._get_sim_description(soup)
                self.snippets.append({
                    "sim_name": self._get_sim_title(soup),
                    "sim_settings": self._get_sim_settings(soup),
                    "sim_desc": sim_desc,
                    "sim_keywords": self.get_sim_keywords(sim_desc),
                    "sim_url": url
                })
                counter += 1
                if counter % 5 == 0:
                    log.debug(f"Currently got {counter} recipes.")
        return self.snippets

    def prep_data(self):
        ss_data = []
        for d in self.get_snippets():
            for chunk in d["sim_desc"].split("\n"):
                if chunk:
                    ss_data.append([d["sim_name"], d["sim_keywords"], chunk])
        return ss_data

    def get_embeddings(self, model_dir):
        model = SentenceTransformer(model_dir)
        lowered_data = [f"passage: {' '.join(i).lower()}" for i in self.prep_data()]
        embeddings = model.encode(lowered_data, convert_to_tensor=True)
        return embeddings

    def get_settings(self):
        return {d["sim_name"]: d["sim_settings"] for d in self.get_snippets()}

    def get_sim_keywords(self, sim_desc):
        r = self.oai_cli.chat.completions.create(
            model=self.kwd_llm,
            messages=[
                {"role": "system", "content": ops.prompts["kwd_generation"]["sys"]},
                {"role": "user", "content": ops.prompts["kwd_generation"]["shot_q"]},
                {"role": "assistant", "content": ops.prompts["kwd_generation"]["shot_a"]},
                {"role": "user", "content": sim_desc}])
        log.debug(sim_desc)
        log.debug(f"generated kwds: {r.choices[0].message.content}")
        return r.choices[0].message.content


if __name__ == "__main__":
    args = ops.parse_my_args([["--target-dir", str, True],
                              ["--debug", bool, False]])

    log.setLevel("DEBUG") if args.debug else log.setLevel("INFO")
    if not os.path.isdir(args.target_dir):
        os.makedirs(args.target_dir)

    # fws = FujiXWeeklyScraper(snippets=DataOps.load_json("data/simulations.json"))
    fws = FujiXWeeklyScraper()
    to_do = [(fws.get_snippets, "simulations"),
             (fws.get_settings, "settings"),
             (fws.prep_data, "snippets")]
    for fun, filename in to_do:
        ops.export_json(fun(), os.path.join(args.target_dir, f"{filename}.json"))

    ops.export_npy(fws.get_embeddings("multilingual-e5-base"), os.path.join(args.target_dir, "embeddings.npy"))
    log.debug("Finished!")
