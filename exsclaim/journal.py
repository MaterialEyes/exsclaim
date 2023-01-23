import itertools
import json
import logging
import math
import os
import pathlib
import random
import shutil
import time
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime

import bs4
import numpy as np
import requests
from dateutil.relativedelta import relativedelta

#try:
from selenium_stealth import stealth
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

#except ImportError:
#    pass
from bs4 import BeautifulSoup

from .utilities import paths


class JournalFamily(ABC):
    """Base class to represent journals and provide scraping methods

    This class defines the interface for interacting with JournalFamilies.
    A JournalFamily is a collection of academic journals with articles
    hosted on a single domain. For example *Nature* is a single journal
    family that serves articles from both *Scientific Reports* and
    *Nature Communications* (and many others) on nature.com.

    The attributes defined mainly consist of documenting the format
    of urls in the journal family. Two urls are of interest:

        * search_results_url: the url one goes to in order to query
          journal articles, and the associated url parameters to filter
          those queries
        * article_url: the general form of the url path containing **html**
          versions of article

    The methods of this class are focused on parsing the html structure
    of the page types returned by the two url types above.

    There are two major types of JournalFamiles (in the future, these
    may make sense to be split into separate subclasses of JournalFamily):
    static and dynamic. Static journal families serve all results using
    static html. Nature is an example. These are simpler as GET requests
    alone will return all of the relevant data. Dynamic familes utilize
    javascript to populate results so a browser emulator like selenium
    is used. RSC is an example of a dynamic journal.

    **Contributing**: If you would like to add a new JournalFamily, decide
    whether a static or dynamic one is needed and look to an existing
    subclass to base your efforts. Create an issue before you start so your
    efforts are not duplicated and submit a PR upon completion. Thanks!
    """

    # journal attributes -- these must be defined for each journal
    # family based on the explanations provided here
    @property
    def domain(self) -> str:
        """The domain name of journal family"""
        return self._domain

    # the next 6 fields determine the url of journals search page
    @property
    def search_path(self) -> str:
        """URL portion from end of top level domain to query parameters"""
        return self._search_path

    # params should include trailing '='
    @property
    def page_param(self) -> str:
        """URL parameter noting current page number"""
        return self._page_param

    @property
    def max_page_size(self) -> str:
        """URL parameter and value requesting max results per page

        Used to limit total number of requests.
        """
        return self._max_page_size

    @property
    def term_param(self) -> str:
        """URL parameter noting search term"""
        return self._term_param

    @property
    def order_param(self) -> str:
        """URL parameter noting results order"""
        return self._order_param

    @property
    def open_param(self) -> str:
        """URL parameter optionally noting open results only"""
        return self._open_param

    @property
    def journal_param(self) -> str:
        """URL paremter noting journal to search"""
        return self._journal_param

    @property
    def date_range_param(self) -> str:
        """URL parameter noting range of dates to search"""
        return self._date_range_param

    @property
    def pub_type(self) -> str:
        """URL paramter noting publication type (specific to Wiley)"""
        return self._pub_type

    @property
    def order_values(self) -> dict:
        """Dictionary with journal parameter values for sorting results

        'relevant' for ordering results in order of relevance
        'recent' for ordering results most recent first
        'old' for ordering results oldest first
        """
        return self._order_values

    @property
    def join(self) -> str:
        """Separator in URL between multiple search terms"""
        return self._join

    @property
    def max_query_results(self) -> int:
        """Maximum results journal family will return for single query

        Certain journals will restrict a given search to ~1000 results
        """
        return self.__max_query_results

    # used for get_article_delimiters
    @property
    def articles_path(self) -> str:
        """The journal's url path to articles

        Articles are located at domain.name/articles_path/article
        """
        return self._articles_path

    @property
    def articles_path_length(self) -> int:
        """Number of / separated segments to articles path"""
        return self._articles_path_length

    def __init__(self, search_query: dict):
        """creates an instance of a journal family search using a query

        Args:
            search_query: a query json (python dictionary)
        Returns:
            An initialized instance of a search on a journal family
        """
        self.search_query = search_query
        self.open = search_query.get("open", False)
        self.order = search_query.get("order", "relevant")
        self.logger = logging.getLogger(__name__)
        # Set up file structure
        base_results_dir = paths.initialize_results_dir(
            self.search_query.get("results_dirs", None)
        )
        self.results_directory = base_results_dir / self.search_query["name"]
        figures_directory = self.results_directory / "figures"
        os.makedirs(figures_directory, exist_ok=True)

        # Check if any articles have already been scraped by checking
        # results_dir/_articles
        articles_visited = {}
        articles_file = self.results_directory / "_articles"
        if os.path.isfile(articles_file):
            with open(articles_file, "r") as f:
                contents = f.readlines()
            articles_visited = {a.strip() for a in contents}
        self.articles_visited = articles_visited

    # Helper Methods for retrieving relevant article URLS

    @abstractmethod
    def get_page_info(self, soup: BeautifulSoup) -> tuple:
        """Retrieve details on total results from search query

        Args:
            soup: a search results page
        Returns:
            (index origin, total page count in search, total results from search)
        """
        pass

    @abstractmethod
    def turn_page(self, url: str, page_number: int) -> BeautifulSoup:
        """Return page_number page of search results

        Args:
            url: the url to a search results page
            page_number: page number to search on
        Returns:
            soup of next page
        """
        new_url = url + "&" + self.page_param + str(page_number)
        return self.get_soup_from_request(new_url)

    @abstractmethod
    def get_additional_url_arguments(self, soup: BeautifulSoup) -> tuple:
        """Get lists of additional search url parameters

        Some JournalFamilies limit the number of articles returned
        by a single search. In order to retrieve articles beyond this,
        we create additional search queries filtering for non-overlapping
        sets, and execute them individually.

        Args:
            soup: initial search result for search term
        Returns:
            (years, journal_codes, orderings): where:
                years is a list of strings of desired date ranges
                journal_codes is a list of strings of desired journal codes
                orderings is a list of strings of desired results ordering
            Each of these should be in order of precedence.
        """
        pass

    # Helper Methods for retrieving figures from articles

    @abstractmethod
    def get_license(self, soup: BeautifulSoup) -> tuple:
        """Checks the article license and whether it is open access

        Args:
            soup: representation of page html
        Returns:
            is_open (a bool): True if article is open
            license (a string): Requried text of article license
        """
        return (False, "unknown")

    @abstractmethod
    def is_link_to_open_article(self, tag: BeautifulSoup) -> bool:
        """Checks if link is to an open access article

        Args:
            tag (bs4.tag): A tag containing an href attribute that
                links to an article
        Returns:
            True if the article is confirmed open_access
        """
        return False

    @abstractmethod
    def get_figure_subtrees(self, soup: BeautifulSoup) -> list:
        """Retrieves list of bs4 parse subtrees containing figure elements

        Args:
            soup: A beautifulsoup parse tree
        Returns:
            A list of all figures in the article as BeaustifulSoup objects
        """
        figure_list = [
            a for a in soup.find_all("figure") if str(a).find(self.extra_key) > -1
        ]
        return figure_list

    @abstractmethod
    def get_soup_from_request(self, url: str) -> BeautifulSoup:
        """Get a BeautifulSoup parse tree (lxml parser) from a url request

        Args:
            url: A requested url
        Returns:
            A BeautifulSoup parse tree.
        """
        headers = {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": (
                "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:82.0)"
                " Gecko/20100101 Firefox/82.0"
            ),
        }
        wait_time = float(random.randint(0, 50))
        time.sleep(wait_time / float(10))
        with requests.Session() as session:
            r = session.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "lxml")
        return soup

    @abstractmethod
    def find_captions(self, figure_subtree: BeautifulSoup) -> bs4.element.ResultSet:
        """
        Returns all captions associated with a given figure

        Args:
            figure_subtree: an bs4 parse tree
        Returns:
            all captions for given figure
        """
        return figure_subtree.find_all("p")

    @abstractmethod
    def save_figure(self, figure_name: str, image_url: str):
        """
        Saves figure at img_url to local machine

        Args:
            figure_name: name of figure
            img_url: url to image
        """
        figures_directory = self.results_directory / "figures"
        response = requests.get(image_url, stream=True)
        figure_path = figures_directory / figure_name
        with open(figure_path, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

    @abstractmethod
    def get_figure_url(self, figure_subtree: BeautifulSoup) -> str:
        """Returns url of figure from figure's html subtree

        Args:
            figure_subtree (bs4): subtree containing an article figure
        Returns:
            url (str)
        """
        image_tag = figure_subtree.find("img")
        image_url = image_tag.get("src")
        return self.prepend + image_url

    def get_search_query_urls(self) -> list:
        """Create list of search query urls based on input query json

        Returns:
            A list of urls (as strings)
        """
        search_query = self.search_query
        # creates a list of search terms
        search_list = [
            [search_query["query"][key]["term"]]
            + search_query["query"][key].get("synonyms", [])
            for key in search_query["query"]
        ]
        search_product = list(itertools.product(*search_list))

        search_urls = []
        for term in search_product:
            url_parameters = "&".join(
                [self.term_param + self.join.join(term), self.max_page_size]
            )
            search_url = self.domain + self.search_path + self.pub_type + url_parameters
            if self.open:
                search_url += "&" + self.open_param + "&"
            soup = self.get_soup_from_request(search_url)
            years, journal_codes, orderings = self.get_additional_url_arguments(soup)
            search_url_args = []

            for year_value in years:
                year_param = self.date_range_param + year_value
                for journal_value in journal_codes:
                    for order_value in orderings:
                        args = "&".join(
                            [
                                year_param,
                                self.journal_param + journal_value,
                                self.order_param + order_value,
                            ]
                        )
                        search_url_args.append(args)
            search_term_urls = [search_url + url_args for url_args in search_url_args]
            search_urls += search_term_urls
        return search_urls

    def get_articles_from_search_url(self, search_url: str) -> list:
        """Generates a list of articles from a single search term"""
        max_scraped = self.search_query["maximum_scraped"]
        self.logger.info("GET request: {}".format(search_url))
        soup = self.get_soup_from_request(search_url)
        start_page, stop_page, total_articles = self.get_page_info(soup)
        article_paths = set()
        for page_number in range(start_page, stop_page + 1):
            for tag in soup.find_all("a", href=True):
                url = tag.attrs["href"]
                self.logger.debug("Candidate Article: {}".format(url))
                if (
                    self.articles_path not in url
                    or url.count("/") != self.articles_path_length
                ):
                    # The url does not point to an article
                    continue
                if url.split("/")[-1] in self.articles_visited or (
                    self.open and not self.is_link_to_open_article(tag)
                ):
                    # It is an article but we are not interested
                    continue
                self.logger.debug("Candidate Article: PASS")
                article_paths.add(url)
                if len(article_paths) >= max_scraped:
                    return article_paths
            # Get next page at end of loop since page 1 is obtained from
            # search_url
            soup = self.turn_page(search_url, page_number + 1)

        return article_paths

    def get_article_extensions(self) -> list:
        """Retrieves a list of article url paths from a search query"""
        # This returns urls based on the combinations of desired search terms.
        search_query_urls = self.get_search_query_urls()
        article_paths = set()
        for search_url in search_query_urls:
            new_article_paths = self.get_articles_from_search_url(search_url)
            article_paths.update(new_article_paths)
            if len(article_paths) >= self.search_query["maximum_scraped"]:
                break
        return list(article_paths)

    def get_license(self, soup):
        """ Checks the article license and whether it is open access 
        Args:
            soup (a BeautifulSoup parse tree): representation of page html
        Returns:
            is_open (a bool): True if article is open
            license (a string): Requried text of article license
        """
        return (False, "unknown")

    def get_soup_from_request(self, url: str, fast_load=True):
        """
        Get a BeautifulSoup parse tree (lxml parser) from a url request 
        Args:
            url: A requested url 
        Returns:
            A BeautifulSoup parse tree.
        """
        headers = {"Accept":   "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                   "Accept-Encoding": "gzip, deflate, br",
                   "Accept-Language": "en-US,en;q=0.5",
                   "Upgrade-Insecure-Requests":   "1",
                   "User-Agent":  "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:82.0) Gecko/20100101 Firefox/82.0"}
        wait_time = float(random.randint(0, 50))
        time.sleep(wait_time/float(10))
        with requests.Session() as session:
            r = session.get(url, headers=headers) 
        soup = BeautifulSoup(r.text, 'lxml')
        return soup

    def get_article_figures(self, url: str) -> dict:
        """Get all figures from an article

        Args:
            url: A url to a journal article
        Returns:
            A dict of figure_jsons from an article
        """
        soup = self.get_soup_from_request(url)
        is_open, license = self.get_license(soup)

        # Uncomment to save html
        html_directory = self.results_directory / "html"
        os.makedirs(html_directory, exist_ok=True)
        with open(
            html_directory / (url.split("/")[-1] + ".html"), "w", encoding="utf-8"
        ) as file:
            file.write(str(soup))

        figure_subtrees = self.get_figure_subtrees(soup)
        self.logger.info(len(figure_subtrees))
        figure_number = 1
        article_json = {}

        for figure_subtree in figure_subtrees:
            captions = self.find_captions(figure_subtree)

            # acs captions are duplicated, one version with no captions
            if len(captions) == 0:
                continue

            # get figure caption
            figure_caption = ""
            for caption in captions:
                figure_caption += caption.get_text()

            image_url = self.get_figure_url(figure_subtree)
            # image_url = self.prepend + image_url.replace('_hi-res','')
            if ":" not in image_url:
                image_url = "https:" + image_url
            article_name = url.split("/")[-1].split("?")[0]
            figure_name = article_name + "_fig" + str(figure_number) + ".jpg"
            figure_path = (
                pathlib.Path(self.search_query["name"]) / "figures" / figure_name
            )
            # initialize the figure's json
            figure_json = {
                "title": soup.find("title").get_text(),
                "article_url": url,
                "article_name": article_name,
                "image_url": image_url,
                "figure_name": figure_name,
                "license": license,
                "open": is_open,
                "full_caption": figure_caption,
                "caption_delimiter": "",
                "figure_path": str(figure_path),
                "master_images": [],
                "unassigned": {
                    "master_images": [],
                    "dependent_images": [],
                    "inset_images": [],
                    "subfigure_labels": [],
                    "scale_bar_labels": [],
                    "scale_bar_lines": [],
                    "captions": [],
                },
            }
            # add all results
            article_json[figure_name] = figure_json
            self.save_figure(figure_name, image_url)
            # increment index
            figure_number += 1
        return article_json

class JournalFamilyDynamic(JournalFamily):

    def __init__(self):
        # initiallize the selenium-stealth 
        options = webdriver.ChromeOptions()
        options.add_argument("start-maximized")  
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome('chromedriver', options=options)

        stealth(self.driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
                )

    def get_article_extensions(self, articles_visited=set()) -> list:
        """
        Create a list of article url extensions from search_query
        Returns:
            A list of article url extensions from search.
        """
        search_query = self.search_query
        maximum_scraped = search_query["maximum_scraped"]
        article_delim, reader_delims = self.get_article_delimiters()
        search_query_urls = self.get_search_query_urls()
        article_paths = set()
        for page1 in search_query_urls:
            self.logger.info("GET request: {}".format(page1))
            start_page, stop_page, total_articles = self.get_page_info(page1)
            for page_number in range(start_page, stop_page + 1):
                request = self.turn_page(page1, page_number, total_articles)    
                self.driver.get(request)
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                for tag in soup.find_all('a', href=True):
                    article = tag.attrs['href']
                    article = article.split('?page=search')[0]

                    if (len(article.split(article_delim)) > 1 
                            and article.split("/")[-1] not in articles_visited
                            and article != None
                            and len(set(reader_delims).intersection(set(article.split("/")))) <= 0
                            and not (self.open
                                     and not self.is_link_to_open_article(tag))):
                        article_paths.add(article)
                    if len(article_paths) >= maximum_scraped:
                        return list(article_paths)
        return list(article_paths)

    def get_article_figures(self, url: str) -> dict:
        """
        Get all figures from an article 
        Args:
            url: A url to a journal article
        Returns:
            A dict of figure_jsons from an article
        """

        self.driver.get(url)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        is_open, license = self.get_license(soup)

        html_directory = self.results_directory / "html"
        os.makedirs(html_directory, exist_ok=True)
        with open(html_directory / (url.split("/")[-1]+'.html'), "w", encoding='utf-8') as file:
            file.write(str(soup))

        figure_list = self.get_figure_list(url)
        figures = 1
        article_json = {}

        # for figure in soup.find_all('figure'):
        for figure in figure_list:
            captions = self.find_captions(figure)

            # acs captions are duplicated, one version with no captions
            if len(captions) == 0:
                continue

            # initialize the figure's json
            article_name = url.split("/")[-1]
            figure_json = {"title": soup.find('title').get_text(), 
                            "article_url" : url,
                            "article_name" : article_name}

            # get figure caption
            figure_caption = ""
            for caption in captions:
                figure_caption += caption.get_text()
            figure_json["full_caption"] = figure_caption

            # Allocate entry for caption delimiter
            figure_json["caption_delimiter"] = ""

            # get figure url and name
            if 'rsc' in url.split("."):
                # for image_tag in figure.find_all("a", href=True):
                for image_tag in [a for a in figure.find_all("a", href=True) if str(a).find(self.extra_key)>-1]:
                    image_url = image_tag['href']
            else:
                image_tag = figure.find('img')
                image_url = image_tag.get('src')

            image_url = self.prepend + image_url.replace('_hi-res','')
            if ":" not in image_url:
                image_url = "https:" + image_url
            figure_name = article_name + "_fig" + str(figures) + ".jpg"  #" +  image_url.split('.')[-1]
            print('fig_name',figure_name)
            print('im_url',image_url)
            # save image info
            figure_json["figure_name"] = figure_name
            figure_json["image_url"] = image_url
            figure_json["license"] = license
            figure_json["open"] = is_open

            # save figure as image
            self.save_figure(figure_name, image_url)
            figure_path = (
                pathlib.Path(self.search_query["name"]) / "figures" / figure_name
            )
            figure_json["figure_path"] = str(figure_path)
            figure_json["master_images"] = []
            figure_json["unassigned"] = {
                'master_images': [],
                'dependent_images': [],
                'inset_images': [],
                'subfigure_labels': [],
                'scale_bar_labels':[],
                'scale_bar_lines': [],
                'captions': []
            }
            # add all results
            article_json[figure_name] = figure_json
            # increment index
            figures += 1
        return article_json

    def get_figure_list(self, url):
        """
        Returns list of figures in the givin url
        Args:
            url: a string, the url to be searched
        Returns:
            A list of all figures in the article as BeaustifulSoup Tag objects
        """

        self.driver.get(url)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        figure_list = [a for a in soup.find_all('figure') if str(a).find(self.extra_key)>-1]
        return figure_list
# ############# JOURNAL FAMILY SPECIFIC INFORMATION ################
# To add a new journal family, create a new subclass of
# JournalFamily. Fill out the methods and attributes according to
# their descriptions in the JournalFamily class. Then add the an
# entry to the journals dictionary with the journal family's name in
# all lowercase as the key and the new class as the value.
# ####################################################################


class ACS(JournalFamily):
    domain = "https://pubs.acs.org"
    search_path = "/action/doSearch?"
    term_param = "AllField="
    max_page_size = "pageSize=100"
    page_param = "startPage="
    order_param = "sortBy="
    open_param = "openAccess=18&accessType=openAccess"
    journal_param = "SeriesKey="
    date_range_param = "Earliest="
    # order options
    order_values = {
        "relevant": "relevancy",
        "old": "Earliest_asc",
        "recent": "Earliest",
    }
    join = '"+"'

    articles_path = "/doi/"
    prepend = "https://pubs.acs.org"
    extra_key = "inline-fig internalNav"
    articles_path_length = 3
    max_query_results = 1000

    def get_page_info(self, soup):
        totalResults = int(soup.find("span", {"class": "result__count"}).text)
        totalPages = math.ceil(float(totalResults) / 20) - 1
        page = 0
        return page, totalPages, totalResults

    def get_additional_url_arguments(self, soup):
        now = datetime.now()
        time_format = "%Y%m%d"
        journal_list = soup.find(id="Publication")
        journal_link_tags = journal_list.parent.find_all("a", href=True)
        journal_codes = [jlt.attrs["href"].split("=")[-1] for jlt in journal_link_tags]
        if self.order == "exhaustive":
            num_years = 100
            orderings = list(self.order_values.values())
        else:
            num_years = 25
            orderings = [self.order_values[self.order]]
            journal_codes = journal_codes[:10]
        years = [
            "[{} TO {}]".format(
                (now - relativedelta(years=k - 1)).strftime(time_format),
                (now - relativedelta(years=k)).strftime(time_format),
            )
            for k in range(1, num_years)
        ]
        years = [""] + years
        return years, journal_codes, orderings

    def get_license(self, soup):
        open_access = soup.find("div", {"class": "article_header-open-access"})
        if open_access and (
            "ACS AuthorChoice" in open_access.text
            or "ACS Editors' Choice" in open_access.text
        ):
            is_open = True
            return (is_open, open_access.text)
        return (False, "unknown")

    def is_link_to_open_article(self, tag):
        # ACS allows filtering for search. Therefore, if self.open is
        # true, all results will be open.
        return self.open


class Nature(JournalFamily):
    domain = "https://www.nature.com"
    search_path = "/search?"
    page_param = "page="
    max_page_size = ""  # not available for nature
    term_param = "q="
    order_param = "order="
    open_param = ""
    date_range_param = "date_range="
    journal_param = "journal="
    pub_type = ""
    # order options
    order_values = {"relevant": "relevance", "old": "date_asc", "recent": "date_desc"}
    # codes for journals most relevant to materials science
    materials_journals = [
        "",
        "nature",
        "nmat",
        "ncomms",
        "sdata",
        "nnano",
        "natrevmats",
        "am",
        "npj2dmaterials",
        "npjcompumats",
        "npjmatdeg",
        "npjquantmats",
        "commsmat",
    ]

    join = " "
    articles_path = "/articles/"
    articles_path_length = 2
    prepend = ""
    extra_key = " "
    max_query_results = 1000

    def find_captions(self, figure_subtree: BeautifulSoup):
        return super().find_captions(figure_subtree)

    def get_figure_subtrees(self, soup: BeautifulSoup) -> list:
        return super().get_figure_subtrees(soup)

    def get_figure_url(self, figure_subtree: BeautifulSoup) -> str:
        return super().get_figure_url(figure_subtree)

    def get_soup_from_request(self, url: str) -> BeautifulSoup:
        return super().get_soup_from_request(url)

    def save_figure(self, figure_name: str, image_url: str):
        return super().save_figure(figure_name, image_url)

    def turn_page(self, url: str, page_number: int) -> BeautifulSoup:
        return super().turn_page(url, page_number)

    def get_page_info(self, soup):
        # Finds total results, page number, and total pages in article html
        # Data exists as json inside script tag with 'data-test'='dataLayer' attr.
        data_layer = soup.find(attrs={"data-test": "results-data"})
        # <span data-test="results-data"><span>Showing  <X>–<X+pg_size>
        # of&nbsp;</span><span>N results</span></span>
        current_page_tag, total_results_tag = data_layer.contents
        result_range = current_page_tag.contents[0].split()[1]
        start, end = result_range.split("–")
        page_size = int(end) - int(start) + 1
        page = int(end) // page_size
        total_results = int(total_results_tag.contents[0].split(" ")[0])
        total_pages = math.ceil(total_results / page_size)
        return (page, total_pages, total_results)

    def get_additional_url_arguments(self, soup):
        current_year = datetime.now().year
        earliest_year = 1845
        non_exhaustive_years = 25
        # If the search is exhaustive, search all 161 nature journals,
        # for all years since 1845, in relevance, oldest, and youngest order.
        if self.order == "exhaustive":
            search_url = "https://www.nature.com/search/advanced"
            advanced_search = self.get_soup_from_request(search_url)
            journal_tags = advanced_search.find_all(name="journal[]")
            journal_codes = [tag.value for tag in journal_tags]
            years = [
                str(year) + "-" + str(year)
                for year in range(current_year, earliest_year, -1)
            ]
            orderings = list(self.order_values.values())
        # If the search is not exhaustive, search the most relevant materials
        # journals, for the past 25 years, in self.order order.
        else:
            journal_codes = self.materials_journals
            years = [
                str(year) + "-" + str(year)
                for year in range(current_year - non_exhaustive_years, current_year)
            ]
            orderings = [self.order_values[self.order]]
        years = [""] + years
        return years, journal_codes, orderings

    def get_license(self, soup):
        data_layer = soup.find(attrs={"data-test": "dataLayer"})
        data_layer_string = str(data_layer.string)
        data_layer_json = (
            "{" + data_layer_string.split("[{", 1)[1].split("}];", 1)[0] + "}"
        )
        parsed = json.loads(data_layer_json)
        # try to get whether the journal is open
        try:
            is_open = parsed["content"]["attributes"]["copyright"]["open"]
        except KeyError:
            is_open = False
        # try to get license
        try:
            license = parsed["content"]["attributes"]["copyright"]["legacy"][
                "webtrendsLicenceType"
            ]
        except KeyError:
            license = "unknown"
        return is_open, license

    def is_link_to_open_article(self, tag):
        current_tag = tag
        while current_tag.parent:
            if (
                current_tag.name == "li"
                and "app-article-list-row__item" in current_tag["class"]
            ):
                break
            current_tag = current_tag.parent
        candidates = current_tag.find_all("span", class_="u-color-open-access")
        for candidate in candidates:
            if candidate.text.startswith("Open"):
                return True
        return False


class RSC(JournalFamilyDynamic):
    domain =        "https://pubs.rsc.org"
    relevant =      "Relevance"
    recent =        "Latest%20to%20oldest"
    path =          "/en/results?searchtext="
    join =          "\"%20\""
    pre_sb =        "\"&SortBy="
    open_pre_sb =   "\"&SortBy="
    post_sb =       "&PageSize=1&tab=all&fcategory=all&filter=all&Article%20Access=Open+Access"
    article_path =  ('/en/content/articlehtml/','')
    prepend =       "https://pubs.rsc.org"
    extra_key =     "/image/article"

    def get_page_info(self, url):
        self.driver.get(url)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        self.driver.close()
        possible_entries = [a.strip("\n") for a in soup.text.split(" - Showing page 1 of")[0].split("Back to tab navigation")[-1].split(" ") if a.strip("\n").isdigit()]
        if len(possible_entries) == 1:
            totalResults = possible_entries[0]
        else:
            totalResults = 0
        totalPages = 1
        page = 1
        return page, totalPages, totalResults

    def turn_page(self, url, page_number, pg_size):
        return url.split('1&tab=all')[0]+str(pg_size)+'&tab=all&fcategory=all&filter=all&Article%20Access=Open+Access'

    def get_figure_list(self, url):
        soup = self.get_soup_from_request(url)
        figure_list = soup.find_all("div", class_="image_table")
        return figure_list

    def find_captions(self, figure):
        return figure.find_all("span", class_="graphic_title")

    def save_figure(self, figure_name, image_url):
        figures_directory = self.results_directory / "figures"
        out_file = figures_directory / figure_name
        urllib.request.urlretrieve(image_url, out_file)

    def get_license(self, soup):
        """ Checks the article license and whether it is open access 
        Args:
            soup (a BeautifulSoup parse tree): representation of page html
        Returns:
            is_open (a bool): True if article is open
            license (a string): Requried text of article license
        """
        return (False, "unknown")

    def get_additional_url_arguments(self, soup):
        # rsc allows unlimited results, so no need for additoinal args
        return [""], [""], [""]

    def is_link_to_open_article(self, tag):
        return self.open
    
    def get_figure_subtrees(self, soup):
        figure_subtrees = soup.find_all("div", "image_table")
        return figure_subtrees

    def get_figure_url(self, figure_subtree):
        return self.prepend + figure_subtree.find("a", href=True)["href"]

    def get_soup_from_request(self, url: str) -> BeautifulSoup:
        return super().get_soup_from_request(url)



class Wiley(JournalFamily):
    domain = "https://onlinelibrary.wiley.com"
    search_path = "/action/doSearch?"
    page_param = "startPage="
    max_page_size = "pageSize=20"
    term_param = "AllField="
    order_param = "sortBy="
    open_param = "ConceptID=15941"
    journal_param = "SeriesKey="
    date_range_param = "AfterYear="
    pub_type = "PubType=journal"  # Type of publication hosted on Wiley
    # order options
    order_values = {"relevant": "relevancy", "recent": "Earliest", "old": ""}
    # join is for terms in search query
    join = '"+"'
    max_query_results = 2000
    articles_path = "/doi/"
    prepend = "https://onlinelibrary.wiley.com"
    extra_key = " "
    articles_path_length = 3

    def get_page_info(self, soup):
        totalResults = soup.find("span", {"class": "result__count"}).text
        totalResults = int(totalResults.replace(",", ""))

        totalPages = math.ceil(float(totalResults / 20)) - 1
        page = 0
        return page, totalPages, totalResults

    def get_additional_url_arguements(self, soup):
        current_year = datetime.now().year
        journal_list = soup.find(id="Published in").parent.next_sibling
        journal_link_tags = journal_list.find_all("a", href=True)
        journal_link_tags_exh = journal_list.find_all("option", value=True)

        journal_codes = [jlt.attrs["href"].split("=")[-1] for jlt in journal_link_tags]

        if self.order == "exhaustive":
            num_years = 100
            orderings = list(self.order_values.values())
            journal_codes = journal_codes + [
                jlte.attrs["value"].split("=")[-1] for jlte in journal_link_tags_exh
            ]

        else:
            num_years = 25
            orderings = [self.order_values[self.order]]
        # the wiley search engine uses 2 different phrases to dilineate
        # start and stop year AfterYear=YYYY&BeforeYear=YYYY
        years = [
            "{}&BeforeYear={}".format(year, year)
            for year in range(current_year - num_years, current_year)
        ]

        years = [["", ""]] + years
        return years, journal_codes, orderings

    def turn_page(self, url, pg_num):
        return (
            url.split("&startPage=")[0]
            + "&startPage="
            + str(pg_num)
            + "&pageSize="
            + str(20)
        )

    def get_license(self, soup):
        open_access = soup.find("div", {"class": {"doi-access"}})
        if open_access and "Open Access" in open_access.text:
            is_open = True
            return (is_open, open_access.text)
        return (False, "unknown")

    def is_link_to_open_article(self, tag):
        # Wiley allows filtering for search. Therefore, if self.open is
        # true, all results will be open.
        return self.open
