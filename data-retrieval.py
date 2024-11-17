import json
import logging
import os
from http.client import HTTPConnection
from pprint import pprint as pp
from random import sample

import chromadb
import requests
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ops import dataops as dto
from ops import nlpops as nlo

log = logging.getLogger("FujiXWeeklyScraper")
HTTPConnection.debuglevel = 0
logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s')


class FujiXWeeklyScraper:
    def __init__(self):
        """
        Scrape the website and prepare data for search.
        """
        self.url_ = "https://fujixweekly.com/sitemap-1.xml"
        self.snippets = []
        self.stopwords = ["Share this", "Nobody pays", "Help Fuji X", "Click to share", "*Related*", "!["]
        self.oai_cli = OpenAI()
        self.nlp = nlo.load_nlp("en_core_web_sm")

    def get_recipe_urls(self) -> list[str]:
        recipe_urls = []
        for link in [i.text for i in self.get_soup(self.url_, "xml").findAll("loc")]:
            if "simulation-recipe-" in link:
                recipe_urls.append(link)
        log.debug(f"Got {len(recipe_urls)} recipe urls")
        return recipe_urls

    def get_page(self, url_: str):
        r = requests.get(url_)
        if not r.ok:
            raise RuntimeError(f"Couldn't retrieve page {url_}")
        return r

    def get_soup(self, url: str, parser="html.parser") -> BeautifulSoup:
        res = self.get_page(url)
        return BeautifulSoup(res.content, parser)

    def _get_sim_title(self, sim_soup: BeautifulSoup) -> str:
        return str(sim_soup.find("h1", {"class": "entry-title"}).text)

    def _get_sim_description(self, sim_soup: BeautifulSoup) -> str:
        description = []
        for i in sim_soup.find("div", {"class": "entry-content"}):
            if i.text:
                if self.contains_verb(i.text) and not any([w in md(str(i)) for w in self.stopwords]):
                    description.append(md(str(i)))
        return " ".join(description)

    def _get_sim_settings(self, sim_soup: BeautifulSoup) -> str:
        settings = [md(str(i)) for i in sim_soup.find("div", {"class": "entry-content"}).findAll("strong")]
        return "\n".join(settings)

    def contains_verb(self, text: str) -> bool:
        for token in self.nlp(text):
            if token.pos_ == "VERB":
                return True
        return False

    def get_snippets(self) -> list[dict[str, str]]:
        if not self.snippets:
            counter = 0
            for url in self.get_recipe_urls():
                soup = self.get_soup(url)

                sim_desc: str = self._get_sim_description(soup)
                self.snippets.append({
                    "sim_name": self._get_sim_title(soup),
                    "sim_settings": self._get_sim_settings(soup),
                    "sim_desc": sim_desc,
                    "sim_keywords": self.get_sim_keywords(sim_desc),
                    "sim_url": url,
                    "sim_llm_desc": self.get_sim_description(soup)})
                counter += 1
                if counter % 5 == 0:
                    log.debug(f"Currently got {counter} recipes.")
        return self.snippets

    def prep_data(self) -> list[list[str]]:
        ss_data = []
        for d in self.get_snippets():
            for chunk in d["sim_desc"].split("\n"):
                if chunk:
                    ss_data.append([d["sim_name"], d["sim_keywords"], chunk])
        return ss_data

    def get_embeddings(self, model_dir: str | os.PathLike):
        model = SentenceTransformer(model_dir)
        lowered_data = [f"passage: {' '.join(i).lower()}" for i in self.prep_data()]
        embeddings = model.encode(lowered_data, convert_to_tensor=True)
        return embeddings

    def get_settings(self) -> dict[str, str]:
        return {d["sim_name"]: d["sim_settings"] for d in self.get_snippets()}

    def get_sim_keywords(self, sim_desc: str, oai_model: str = "gpt-4o-mini") -> str:
        r = self.oai_cli.chat.completions.create(
            model=oai_model,
            messages=[
                {"role": "system", "content": nlo.prompts["kwd_generation"]["sys"]},
                {"role": "user", "content": nlo.prompts["kwd_generation"]["shot_q"]},
                {"role": "assistant", "content": nlo.prompts["kwd_generation"]["shot_a"]},
                {"role": "user", "content": sim_desc}])
        log.debug(sim_desc)
        log.debug(f"generated kwds: {r.choices[0].message.content}")
        return r.choices[0].message.content

    def get_llm_img_description(self, photo_urls: list):
        img_content = [{"type": "text", "text": nlo.prompts["photo_description"]["sys"]}]
        for url in photo_urls:
            img_content.append({"type": "image_url",
                                "image_url": {"url": url}})
        r = self.oai_cli.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": img_content
                 }])
        return r.choices[0].message.content

    def get_photo_urls(self, soup):
        imgs = soup.findAll("img", {"data-image-meta": True})
        imgs_with_iso = [
            img["data-medium-file"] for img in imgs
            if "iso" in json.loads(img["data-image-meta"])]
        return imgs_with_iso

    def get_sim_description(self, soup, limit_pic_examples_to: int | None = 2):
        img_urls = self.get_photo_urls(soup)
        if limit_pic_examples_to:
            img_urls = sample(img_urls, limit_pic_examples_to)
        description = self.get_llm_img_description(img_urls)
        return description


if __name__ == "__main__":
    args = dto.parse_my_args([["--target-dir", str, False],
                              ["--debug", bool, False]])

    log.setLevel("DEBUG") if args.debug else log.setLevel("INFO")

    fws = FujiXWeeklyScraper()
    to_do = [(fws.get_snippets, "simulations"),
             (fws.get_settings, "settings"),
             (fws.prep_data, "snippets")]
    for fun, filename in to_do:
        dto.export_json(fun(), os.path.join(args.target_dir, f"{filename}.json"))

    # TODO: ...
    # # dto.export_npy(fws.get_embeddings("multilingual-e5-base"), os.path.join(args.target_dir, "embeddings.npy"))
    #
    # if args.target_dir == "database":
    #     chroma_cli = chromadb.Client()
    #     sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    #         model_name="multilingual-e5-base")
    #
    #     collection = chroma_cli.get_or_create_collection(name="recipe-data", embedding_function=sentence_transformer_ef)
    #     collection.add()


    if not os.path.isdir(args.target_dir):
        os.makedirs(args.target_dir)


    log.debug("Finished!")
