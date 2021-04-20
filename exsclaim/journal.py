import os
import ast
import math
import time
import shutil
import urllib.request
import requests
import itertools
import numpy as np
import json
import random
import time
import logging
import pathlib
from .utilities import paths


from bs4 import BeautifulSoup


class JournalFamily():
    """
    Base class to represent journals and provide scraping methods
    """
    ## journal attributes -- these must be defined for each journal
    ## family based on the explanations provided here
    domain =        "The domain name of journal family",
    ## the next 6 fields determine the url of journals search page
    relevant =      ("The value for the journal's sorting field"
                    "that donotes sorting search results based" 
                    "on relevance in the search page url"),
    recent =        "The sorting field value for recency",
    path =          ("The portion of the url that runs from the"
                    "end of the domain to search terms"),
    join =          ("The portion of the url that appears between"
                    "consecutive search terms"),
    pre_sb =        ("Portion of the url that extends from the"
                    "search terms to the paremeter value for"
                    "sorting search results"),
    open_pre_sb =   ("Pre sb, except with parameter (if it exists)"
                     " to only search for open access articles"),
    post_sb =       "The final portion of the url", 
    ## used for get_article_delimiters
    article_path =  "The journal's url path to articles"

    def __init__(self, search_query):
        """
        Initializes an instance of a journal family search using a query

        Args:
            search_query: a query json (python dictionary)
        Returns:
            An initialized instance of a search on a journal family
        """
        self.search_query = search_query
        self.open = search_query.get("open", False)
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

    def get_domain_name(self) -> str:
        """
        Get url base path (domain name) for the requested journal family in search_query.

        Returns:
            The domain name of the journal, as a string
        """
        return self.domain

    def get_article_delimiters(self) -> tuple:
        """
        Get delimiters (url path to articles) specific to this journal

        Returns:
            (delimiter for articles in the journal family url, \
            list containing delimiters for possible reader versions for an article).
        """
        return self.article_path

    def get_page_info(self, soup):
        """
        Get index origin and total number of pages from search_query.

        Args:
            soup: a beautifulSoup object representing the html page of interest

        Returns:
            (index origin, total page count in search, total results from search)
        """
        print("implement get_page_info for journal")

    def turn_page(self, url, pg_num, pg_size):
        """
        Create url for a GET request based on the search_query.

        Args:
            url: the url to a search results page
            pg_num: page number to search on (-1 = starts at index origin for journal)
            pg_size: number of results on given page
        Returns:
            A url.
        """ 
        print("implement get_page_info for journal")

    def get_search_query_urls(self) -> str:
        """
        Create url for a GET request based on the search_query.

        Returns:
            A list of urls (as strings)
        """
        search_query = self.search_query
        ## creates a list of search terms
        search_list = ([[search_query['query'][key]['term']] + 
                       search_query['query'][key]['synonyms'] 
                       for key in search_query['query']])
        search_product = list(itertools.product(*search_list))

        # sortby is the requested method to sort results (relevancy or recency) and
        # sbtext gives the journal specific parameter to sort as requested
        sortby = search_query['sortby']
        if sortby == 'relevant':
            sbext = self.relevant
        elif sortby == 'recent':
            sbext = self.recent

        # modify search to find only open access articles
        if search_query.get("open", False):
            self.pre_sb = self.open_pre_sb

        search_query_urls = []
        # this creates the url (excluding domain and protocol) for each search query
        for search_group in search_product:
            search_query_url = (self.domain + self.path
                + self.join.join(["+".join(a.split(" ")) for a in search_group])
                + self.pre_sb + sbext + self.post_sb)
            search_query_urls.append(search_query_url)

        return search_query_urls

    def get_license(self, soup):
        """ Checks the article license and whether it is open access 

        Args:
            soup (a BeautifulSoup parse tree): representation of page html
        Returns:
            is_open (a bool): True if article is open
            license (a string): Requried text of article license
        """
        return (False, "unknown")

    def is_link_to_open_article(self, tag):
        """ Checks if link is to an open access article 
        
        Args:
            tag (bs4.tag): A tag containing an href attribute that
                links to an article
        Returns:
            True if the article is confirmed open_access 
        """
        return False

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
            soup = self.get_soup_from_request(page1, fast_load=True)
            start_page, stop_page, total_articles = self.get_page_info(soup)
            for page_number in range(start_page, stop_page + 1):
                request = self.turn_page(page1, page_number, total_articles)
                soup = self.get_soup_from_request(request, fast_load=False)
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

    def get_figure_list(self, url):
        """
        Returns list of figures in the givin url

        Args:
            url: a string, the url to be searched
        Returns:
            A list of all figures in the article as BeaustifulSoup Tag objects
        """
        soup = self.get_soup_from_request(url)
        figure_list = [a for a in soup.find_all('figure') if str(a).find(self.extra_key)>-1]
        return figure_list

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

    def find_captions(self, figure):
        """
        Returns all captions associated with a given figure

        Args:
            figure: an html figure
        Returns:
            all captions for given figure
        """
        return figure.find_all('p')

    def save_figure(self, figure_name, image_url):
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

    def get_article_figures(self, url: str) -> dict:
        """
        Get all figures from an article 

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


################ JOURNAL FAMILY SPECIFIC INFORMATION ################
## To add a new journal family, create a new subclass of 
## JournalFamily. Fill out the methods and attributes according to
## their descriptions in the JournalFamily class. Then add the an
## entry to the journals dictionary with the journal family's name in
## all lowercase as the key and the new class as the value.
#####################################################################

class ACS(JournalFamily):
    domain =        "https://pubs.acs.org"
    relevant =      "relevancy"
    recent =        "Earliest"
    path =          "/action/doSearch?AllField=\""
    join =          "\"+\""
    pre_sb =        "\"&publication=&accessType=allContent&Earliest=&pageSize=20&startPage=0&sortBy="
    open_pre_sb =   "\"&publication=&openAccess=18&accessType=openAccess&Earliest=&pageSize=20&startPage=0&sortBy="
    post_sb =       ""
    article_path =  ('/doi/',['abs','full','pdf'])
    prepend =       "https://pubs.acs.org"
    extra_key =     "inline-fig internalNav"

    def get_page_info(self, soup):
        totalResults = int(soup.find('span', {'class': "result__count"}).text)
        totalPages = math.ceil(float(totalResults)/20)-1
        page = 0
        return page, totalPages, totalResults

    def turn_page(self, url, pg_num, pg_size):
        return url.split('&startPage=')[0]+'&startPage='+str(pg_num)+"&pageSize="+str(20)

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
    domain =        "https://www.nature.com"
    relevant =      "relevance"
    recent =        "date_desc"
    path =          "/search?q=\""
    join =          "\"%20\""
    pre_sb =        "\"&order="
    open_pre_sb =   "\"&order="
    post_sb =       "&page=1"
    article_path =  ('/articles/','')
    prepend =       ""
    extra_key =     " "

    def get_page_info(self, soup):
        ## Finds total results, page number, and total pages in article html
        ## Data exists as json inside script tag with 'data-test'='dataLayer' attr.
        data_layer = soup.find(attrs = {'data-test': 'dataLayer'})
        data_layer_string = str(data_layer.string)
        data_layer_json = "{" + data_layer_string.split("[{", 1)[1].split("}];", 1)[0] + "}"
        parsed = json.loads(data_layer_json)
        search_info = parsed["page"]["search"]
        return (search_info["page"],
                search_info["totalPages"], 
                search_info["totalResults"])

    def turn_page(self, url, pg_num, pg_size):
        return url.split('&page=')[0]+'&page='+str(pg_num)

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
        i = 0
        current_tag = tag
        while current_tag.parent and i < 3:
            current_tag = current_tag.parent
            i += 1
        candidates = current_tag.find_all("span", class_="text-orange")
        for candidate in candidates:
            if candidate.text == "Open":
                return True
        return False
        

class RSC(JournalFamily):
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

    def get_page_info(self, soup):
        possible_entries = [a.strip("\n") for a in soup.text.split(" - Showing page 1 of")[0].split("Back to tab navigation")[-1].split(" ") if a.strip("\n").isdigit()]
        if len(possible_entries) == 1:
            totalResults = possible_entries[0]
        else:
            totalResults = 0
        totalPages = 1
        page = 1
        return page, totalPages, totalResults

    def turn_page(self, url, pg_num, pg_size):
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