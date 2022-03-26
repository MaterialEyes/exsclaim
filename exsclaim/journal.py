import os
import ast
import math
import time
import shutil
import urllib.request
import bs4
import requests
import itertools
import numpy as np
import json
import random
import time
import logging
import pathlib
from datetime import datetime
from dateutil.relativedelta import relativedelta
from abc import ABC, abstractmethod
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException
    from webdriver_manager.chrome import ChromeDriverManager

except:
    pass
from .utilities import paths


from bs4 import BeautifulSoup


class JournalFamily(ABC):
    """
    Base class to represent journals and provide scraping methods
    """
    ## journal attributes -- these must be defined for each journal
    ## family based on the explanations provided here
    @property
    def domain(self) -> str:
        """ The domain name of journal family """
        return self._domain

    ## the next 6 fields determine the url of journals search page
    @property
    def search_path(self) -> str:    
        """ URL portion from end of TLD to query parameters """
        return self._search_path

    ## params should include trailing '='
    @property
    def page_param(self) -> str:       
        """ URL parameter noting current page number """
        return self._page_param
    
    @property
    def max_page_size(self) -> str:  
        """ URL parameter and value requesting max results per page
        
        Used to limit total number of requests.
        """
        return self._max_page_size
    
    @property
    def term_param(self) -> str:       
        """ URL parameter noting search term """
        return self._term_param
    
    @property
    def order_param(self) -> str:      
        """ URL parameter noting results order """
        return self._order_param
    
    @property
    def open_param(self) -> str:       
        """ URL parameter optionally noting open results only """
        return self._open_param
    
    @property
    def journal_param(self) -> str:    
        """ URL paremter noting journal to search """
        return self._journal_param
    
    @property
    def date_range_param(self) -> str: 
        """ URL parameter noting range of dates to search """
        return self._date_range_param

    @property
    def pub_type(self) -> str:       
        """ URL paramter noting publication type (specific to Wiley) """
        return self._pub_type
    
    @property
    def order_values(self) -> dict:
        """ Dictionary with journal parameter values for sorting results 
        
        'relevant' for ordering results in order of relevance
        'recent' for ordering results most recent first
        'old' for ordering results oldest first
        """
        return self._order_values
    
    @property
    def join(self) -> str:
        """ Separator in URL between multiple search terms """
        return self._join

    @property
    def max_query_results(self) -> int:
        """ Maximum results journal family will return for single query

        Certain journals will restrict a given search to ~1000 results
        """
        return self.__max_query_results

    ## used for get_article_delimiters
    @property
    def articles_path(self) -> str:
        """ The journal's url path to articles

        Articles are located at domain.name/articles_path/article
        """
        return self._articles_path

    @property
    def articles_path_length(self) -> int:
        """ Number of / separated segments to articles path """
        return self._articles_path_length

    def __init__(self, search_query):
        """ creates an instance of a journal family search using a query

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
        self.results_directory = (
            base_results_dir / self.search_query["name"]
        )
        figures_directory = self.results_directory / "figures"
        os.makedirs(figures_directory, exist_ok=True)

        ## Check if any articles have already been scraped by checking
        ## results_dir/_articles
        articles_visited = {}
        articles_file = self.results_directory / "_articles"
        if os.path.isfile(articles_file):
            with open(articles_file,'r') as f:
                contents = f.readlines()
            articles_visited ={a.strip() for a in contents}
        self.articles_visited = articles_visited

    ## Helper Methods for retrieving relevant article URLS

    @abstractmethod
    def get_page_info(self, soup: bs4.BeautifulSoup) -> tuple:
        """ Retrieve details on total results from search query

        Args:
            soup: a search results page
        Returns:
            (index origin, total page count in search, total results from search)
        """
        pass

    @abstractmethod
    def turn_page(self, url: str, page_number: int) -> bs4.BeautifulSoup:
        """ Return page_number page of search results

        Args:
            url: the url to a search results page
            page_number: page number to search on
        Returns:
            soup of next page
        """ 
        new_url = url + "&" + self.page_param + str(page_number)
        return self.get_soup_from_request(new_url)

    @abstractmethod
    def get_additional_url_arguments(self, soup: bs4.BeautifulSoup) -> tuple:
        """ Get lists of additional search url parameters

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

    ## Helper Methods for retrieving figures from articles

    @abstractmethod
    def get_license(self, soup: bs4.BeautifulSoup) -> tuple:
        """ Checks the article license and whether it is open access

        Args:
            soup: representation of page html
        Returns:
            is_open (a bool): True if article is open
            license (a string): Requried text of article license
        """
        return (False, "unknown")

    @abstractmethod
    def is_link_to_open_article(self, tag: bs4.BeautifulSoup) -> bool:
        """ Checks if link is to an open access article

        Args:
            tag (bs4.tag): A tag containing an href attribute that
                links to an article
        Returns:
            True if the article is confirmed open_access
        """
        return False

    @abstractmethod
    def get_figure_subtrees(self, soup: bs4.BeautifulSoup) -> list:
        """ Retrieves list of bs4 parse subtrees containing figure elements

        Args:
            soup: A beautifulsoup parse tree
        Returns:
            A list of all figures in the article as BeaustifulSoup objects
        """
        figure_list = [a for a in soup.find_all('figure') if str(a).find(self.extra_key)>-1]
        return figure_list

    @abstractmethod
    def get_soup_from_request(self, url: str) -> bs4.BeautifulSoup:
        """ Get a BeautifulSoup parse tree (lxml parser) from a url request

        Args:
            url: A requested url
        Returns:
            A BeautifulSoup parse tree.
        """
        headers = {"Accept":   "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                   "Accept-Language": "en-US,en;q=0.5",
                   "Upgrade-Insecure-Requests":   "1",
                   "User-Agent":  "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:82.0) Gecko/20100101 Firefox/82.0"}
        wait_time = float(random.randint(0, 50))
        time.sleep(wait_time/float(10))
        with requests.Session() as session:
            r = session.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'lxml')
        return soup

    @abstractmethod
    def find_captions(self, figure_subtree: bs4.BeautifulSoup):
        """
        Returns all captions associated with a given figure

        Args:
            figure_subtree: an bs4 parse tree
        Returns:
            all captions for given figure
        """
        return figure_subtree.find_all('p')

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
        with open(figure_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

    @abstractmethod
    def get_figure_url(self, figure_subtree: bs4.BeautifulSoup) -> str:
        """ Returns url of figure from figure's html subtree
        
        Args:
            figure_subtree (bs4): subtree containing an article figure
        Returns:
            url (str)
        """
        image_tag = figure_subtree.find('img')
        image_url = image_tag.get('src')
        return self.prepend + image_url

    def get_search_query_urls(self) -> list:
        """ Create list of search query urls based on input query

        Returns:
            A list of urls (as strings)
        """
        search_query = self.search_query
        ## creates a list of search terms
        search_list = ([[search_query['query'][key]['term']] + 
                       search_query['query'][key].get('synonyms', []) 
                       for key in search_query['query']])
        search_product = list(itertools.product(*search_list))

        search_urls = []
        for term in search_product:
            url_parameters = "&".join(
                [self.term_param + self.join.join(term),
                 self.max_page_size]
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
                                self.order_param + order_value
                            ]
                        )
                        search_url_args.append(args)
            search_term_urls = [search_url + url_args for url_args in search_url_args]
            search_urls += search_term_urls
        return search_urls

    def get_articles_from_search_url(self, search_url: str) -> list:
        """ Generates a list of articles from a single search query """
        max_scraped = self.search_query["maximum_scraped"]
        self.logger.info("GET request: {}".format(search_url))
        soup = self.get_soup_from_request(search_url)
        start_page, stop_page, total_articles = self.get_page_info(soup)
        article_paths = set()
        for page_number in range(start_page, stop_page+1):
            for tag in soup.find_all('a', href=True):
                url = tag.attrs['href']
                self.logger.debug("Candidate Article: {}".format(url))
                if (self.articles_path not in url
                      or url.count("/") != self.articles_path_length):
                    # The url does not point to an article
                    continue
                if (url.split("/")[-1] in self.articles_visited
                      or (self.open and not self.is_link_to_open_article(tag))):
                    # It is an article but we are not interested
                    continue
                self.logger.debug("Candidate Article: PASS")
                article_paths.add(url)
                if len(article_paths) >= max_scraped:
                    return article_paths
            # Get next page at end of loop since page 1 is obtained from
            # search_url
            soup = self.turn_page(search_url, page_number+1)

        return article_paths

    def get_article_extensions(self) -> list:
        """ Retrieves a list of article url paths from search_query """
        # This returns urls based on the combinations of desired search terms.
        search_query_urls = self.get_search_query_urls()
        article_paths = set()
        for search_url in search_query_urls:
            new_article_paths = self.get_articles_from_search_url(search_url)
            article_paths.update(new_article_paths)
            if len(article_paths) >= self.search_query["maximum_scraped"]:
                break
        return list(article_paths)

    def get_article_figures(self, url: str) -> dict:
        """ Get all figures from an article

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
        with open(html_directory / (url.split("/")[-1]+'.html'), "w", encoding='utf-8') as file:
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
                "title": soup.find('title').get_text(), 
                "article_url" : url,
                "article_name" : article_name,
                "image_url": image_url,
                "figure_name": figure_name,
                "license": license,
                "open": is_open,
                "full_caption": figure_caption,
                "caption_delimiter": "",
                "figure_path": str(figure_path),
                "master_images": [],
                "unassigned": {
                    'master_images': [],
                    'dependent_images': [],
                    'inset_images': [],
                    'subfigure_labels': [],
                    'scale_bar_labels':[],
                    'scale_bar_lines': [],
                    'captions': []
                }
            }
            # add all results
            article_json[figure_name] = figure_json
            self.save_figure(figure_name, image_url)
            # increment index
            figure_number += 1
        return article_json


################ JOURNAL FAMILY SPECIFIC INFORMATION ################
## To add a new journal family, create a new subclass of
## JournalFamily. Fill out the methods and attributes according to
## their descriptions in the JournalFamily class. Then add the an
## entry to the journals dictionary with the journal family's name in
## all lowercase as the key and the new class as the value.
#####################################################################

class ACS(JournalFamily):
    domain =                "https://pubs.acs.org"
    search_path =           "/action/doSearch?"
    term_param =            "AllField="
    max_page_size =         "pageSize=100"
    page_param =            "startPage="
    order_param =           "sortBy="
    open_param =            "openAccess=18&accessType=openAccess"
    journal_param =         "SeriesKey="
    date_range_param =      "Earliest="
    # order options
    order_values = {
        "relevant" : "relevancy",
        "old" : "Earliest_asc",
        "recent" : "Earliest"
    }
    join =                  "\"+\""

    articles_path =         "/doi/"
    prepend =               "https://pubs.acs.org"
    extra_key =             "inline-fig internalNav"
    articles_path_length =  3
    max_query_results =     1000

    def get_page_info(self, soup):
        totalResults = int(soup.find('span', {'class': "result__count"}).text)
        totalPages = math.ceil(float(totalResults)/20)-1
        page = 0
        return page, totalPages, totalResults

    def get_additional_url_arguments(self, soup):
        now = datetime.now()
        time_format = "%Y%m%d"
        journal_list = soup.find(id="Publication")
        journal_link_tags = journal_list.parent.find_all('a', href=True)
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
                (now - relativedelta(years=k-1)).strftime(time_format),
                (now - relativedelta(years=k)).strftime(time_format)
            ) for k in range(1, num_years)
        ]
        years = [""] + years
        return years, journal_codes, orderings


    def get_license(self, soup):
        open_access = soup.find('div', {"class": "article_header-open-access"})
        if open_access and ("ACS AuthorChoice" in open_access.text or
                            "ACS Editors' Choice" in open_access.text):
            is_open = True
            return (is_open, open_access.text)
        return (False, "unknown")

    def is_link_to_open_article(self, tag):
        # ACS allows filtering for search. Therefore, if self.open is
        # true, all results will be open.
        return self.open


class Nature(JournalFamily):
    domain =                "https://www.nature.com"
    search_path =           "/search?"
    page_param =            "page="
    max_page_size =         "" # not available for nature
    term_param =            "q="
    order_param =           "order="
    open_param =            ""
    date_range_param =      "date_range="
    journal_param =         "journal="
    pub_type =              ""
    # order options
    order_values = {
        "relevant" : "relevance",
        "old" : "date_asc",
        "recent" : "date_desc"
    }
    # codes for journals most relevant to materials science
    materials_journals =    [
        "", "nature", "nmat", "ncomms", "sdata", "nnano", "natrevmats", "am",
        "npj2dmaterials", "npjcompumats", "npjmatdeg", "npjquantmats",
        "commsmat"
    ]

    join =                  " "
    articles_path =         '/articles/'
    articles_path_length =  2
    prepend =       ""
    extra_key =     " "
    max_query_results =     1000

    def find_captions(self, figure_subtree: bs4.BeautifulSoup):
        return super().find_captions(figure_subtree)

    def get_figure_subtrees(self, soup: bs4.BeautifulSoup) -> list:
        return super().get_figure_subtrees(soup)
    
    def get_figure_url(self, figure_subtree: bs4.BeautifulSoup) -> str:
        return super().get_figure_url(figure_subtree)

    def get_soup_from_request(self, url: str) -> bs4.BeautifulSoup:
        return super().get_soup_from_request(url)

    def save_figure(self, figure_name: str, image_url: str):
        return super().save_figure(figure_name, image_url)

    def turn_page(self, url: str, page_number: int) -> bs4.BeautifulSoup:
        return super().turn_page(url, page_number)


    def get_page_info(self, soup):
        ## Finds total results, page number, and total pages in article html
        ## Data exists as json inside script tag with 'data-test'='dataLayer' attr.
        data_layer = soup.find(attrs = {'data-test': 'results-data'})
        # <span data-test="results-data"><span>Showing  <X>–<X+pg_size> of&nbsp;</span><span>N results</span></span>
        current_page_tag, total_results_tag = data_layer.contents
        result_range = current_page_tag.contents[0].split()[1]
        start, end = result_range.split("–")
        page_size = int(end) - int(start) + 1
        page = int(end) // page_size
        total_results = int(total_results_tag.contents[0].split(" ")[0])
        total_pages = math.ceil(total_results / page_size)
        return (page,
                total_pages,
                total_results)

    def get_additional_url_arguments(self, soup):
        current_year = datetime.now().year
        earliest_year = 1845
        non_exhaustive_years = 25
        ## If the search is exhaustive, search all 161 nature journals,
        ## for all years since 1845, in relevance, oldest, and youngest order.
        if self.order == "exhaustive":
            search_url = "https://www.nature.com/search/advanced"
            advanced_search = self.get_soup_from_request(search_url)
            journal_tags = advanced_search.find_all(name="journal[]")
            journal_codes = [tag.value for tag in journal_tags]
            years = [
                str(year)+"-"+str(year)
                for year in range(current_year, earliest_year, -1)
            ]
            orderings = list(self.order_values.values())
        ## If the search is not exhaustive, search the most relevant materials
        ## journals, for the past 25 years, in self.order order.
        else:
            journal_codes = self.materials_journals
            years = [
                str(year)+"-"+str(year)
                for year in range(current_year-non_exhaustive_years, current_year)
            ]
            orderings = [self.order_values[self.order]]
        years = [""] + years
        return years, journal_codes, orderings

    def get_license(self, soup):
        data_layer = soup.find(attrs = {'data-test': 'dataLayer'})
        data_layer_string = str(data_layer.string)
        data_layer_json = "{" + data_layer_string.split("[{", 1)[1].split("}];", 1)[0] + "}"
        parsed = json.loads(data_layer_json)
        ## try to get whether the journal is open
        try:
            is_open = parsed["content"]["attributes"]["copyright"]["open"]
        except:
            is_open = False
        ## try to get license
        try:
            license = parsed["content"]["attributes"]["copyright"]["legacy"]["webtrendsLicenceType"]
        except:
            license = "unknown"
        return is_open, license

    def is_link_to_open_article(self, tag):
        current_tag = tag
        while current_tag.parent:
            if (current_tag.name == "li"
                and "app-article-list-row__item" in current_tag["class"]):
                break
            current_tag = current_tag.parent
        candidates = current_tag.find_all("span", class_="u-color-open-access")
        for candidate in candidates:
            if candidate.text.startswith("Open"):
                return True
        return False


class RSC(JournalFamily):
    domain =                "https://pubs.rsc.org"
    search_path =           "/en/results?"
    page_param =            ""  # pagination through javascript
    max_page_size =         "PageSize=1000"
    term_param =            "searchtext="
    order_param =           "SortBy="
    open_param =            "Article Access=Open+Access"
    date_range_param =      "Date Range="
    journal_param =         "Journal="
    # order options
    order_values = {
        "relevant" : "Relevance",
        "old" : "Oldest to latest",
        "recent" : "Latest to oldest"
    }
    # codes for journals most relevant to materials science
    materials_journals =    [
        "", "Nanoscale", "RSC+Adv.", "Chem.+Sci.", "Mater.+Adv.",
        "Nanoscale+Adv.", "Mater.+Chem.+Front."
    ]

    join =                  " "
    articles_path =         '/en/content/articlehtml/' # YYYY/2 char journal/
    articles_path_length =  6
    prepend =               "https://pubs.rsc.org"
    extra_key =     " "
    max_query_results =     np.inf

    def __init__(self, search_query):
        super().__init__(search_query)
        ## set up selenium
        chromeOptions = webdriver.ChromeOptions() 
        chromeOptions.add_argument("--no-sandbox")
        chromeOptions.add_argument("--headless")
        chromeOptions.add_argument("--disable-dev-shm-usage") 
        chromeOptions.add_argument("--hide-scrollbars")
        chromeOptions.add_argument('--disable-extensions')
        chromeOptions.add_argument('--profile-directory=Default')
        chromeOptions.add_argument("--incognito")
        chromeOptions.add_argument("--disable-plugins-discovery")
        chromeOptions.add_argument("--start-maximized")
        self.browser = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=chromeOptions)

    def get_soup_from_request(self, url: str) -> bs4.BeautifulSoup:
        url.replace(" ", "+")
        self.browser.get(url)
        article_request = "en/content/articlehtml" in url
        try:
            if article_request:
                element = WebDriverWait(self.browser, 5).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "imgHolder"))
                )
            else:
                element = WebDriverWait(self.browser,30).until(
                    lambda driver: 
                        (driver.find_element(By.XPATH, "//*[@class='capsule__action']") or 
                        driver.find_element(By.CLASS_NAME,'img-tbl__image'))
                )
        except TimeoutException:
            article_msg = "\n  3. Article contains no figures"
            error_msg = (
                "Selenium request unsuccessful for {}. Either\n"
                "  1. internet connection is unstable / slow\n"
                "  2. HTML structure of articles has changed"
                "{}"
            ).format(url, article_msg if article_request else "")
            self.logger.info(error_msg)
        soup = BeautifulSoup(self.browser.page_source, 'lxml')
        return soup

    def get_additional_url_arguments(self, soup):
        # rsc allows unlimited results, so no need for additoinal args
        return [""], [""], [""]

    def is_link_to_open_article(self, tag):
        return self.open

    def get_page_info(self, soup):
        possible_entries = [a.strip("\n") for a in soup.text.split(" - Showing page 1 of")[0].split("Back to tab navigation")[-1].split(" ") if a.strip("\n").isdigit()]
        if len(possible_entries) == 1:
            totalResults = possible_entries[0]
        else:
            totalResults = 0
        totalPages = 1
        page = 1
        return page, totalPages, totalResults

    def turn_page(self, url, pg_num):
        self.get_soup_from_request(url)
        next_page = self.browser.find_element_by_class_name("paging__btn--next")
        self.browser.execute_script("arguments[0].setAttribute('data-pageno',arguments[1])",next_page, pg_num)
        next_page.click()
        soup = BeautifulSoup(self.browser.page_source, 'lxml')
        return soup

    def get_figure_subtrees(self, soup):
        figure_subtrees = soup.find_all("div", "image_table")
        return figure_subtrees

    def get_figure_url(self, figure_subtree):
       return self.prepend + figure_subtree.find("a", href=True)['href']

    def find_captions(self, figure):
        return figure.find_all("span", class_="graphic_title")

    def save_figure(self, figure_name, image_url):
        figures_directory = self.results_directory / "figures"
        out_file = figures_directory / figure_name
        urllib.request.urlretrieve(image_url, out_file)


class Wiley(JournalFamily):
    domain =            "https://onlinelibrary.wiley.com"
    search_path =       "/action/doSearch?"
    page_param =        "startPage="
    max_page_size =     "pageSize=20"
    term_param =        "AllField="
    order_param =       "sortBy="
    open_param =        "ConceptID=15941"
    journal_param =     "SeriesKey="
    date_range_param =  "AfterYear="
    pub_type =          "PubType=journal" # Type of publication hosted on Wiley
    # order options
    order_values = {
        "relevant" :    "relevancy",
        "recent" :      "Earliest",
        "old" :         ""
    }
    # join is for terms in search query
    join =              "\"+\""
    max_query_results = 2000
    articles_path =  "/doi/"
    prepend = "https://onlinelibrary.wiley.com"
    extra_key = " "
    articles_path_length = 3

    def get_page_info(self, soup):
        totalResults = (soup.find('span', {'class': 'result__count'}).text)
        totalResults = int(totalResults.replace(",",""))

        totalPages = math.ceil(float(totalResults/20))-1
        page = 0
        return page, totalPages, totalResults

    def get_additional_url_arguements(self,soup):
        current_year = datetime.now().year
        journal_list = soup.find(id="Published in").parent.next_sibling
        journal_link_tags = journal_list.find_all('a', href=True)
        journal_link_tags_exh = journal_list.find_all('option', value=True)

        journal_codes = [jlt.attrs['href'].split('=')[-1] for jlt in journal_link_tags]


        if self.order == 'exhaustive':
            num_years = 100
            orderings = list(self.order_values.values())
            journal_codes = journal_codes +\
                            [jlte.attrs['value'].split('=')[-1] for jlte in journal_link_tags_exh]

        else:
            num_years = 25
            orderings = [self.order_values[self.order]]
        # the wiley search engine uses 2 different phrases to dilineate
        # start and stop year AfterYear=YYYY&BeforeYear=YYYY
        years = [
            "{}&BeforeYear={}".format(year,year)
            for year in range(current_year-num_years, current_year)
        ]

        years = [["", ""]] + years
        return years, journal_codes, orderings

    def turn_page(self, url, pg_num):
        return url.split('&startPage=')[0]+'&startPage='+str(pg_num)+"&pageSize="+str(20)


    def get_license(self, soup):
        open_access = soup.find('div', {'class':{'doi-access'}})
        if open_access and 'Open Access' in open_access.text:
            is_open = True
            return (is_open, open_access.text)
        return (False, "unknown")

    def is_link_to_open_article(self, tag):
        # Wiley allows filtering for search. Therefore, if self.open is
        # true, all results will be open.
        return self.open
