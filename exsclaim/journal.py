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

from bs4 import BeautifulSoup
from collections import OrderedDict
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options


################ JOURNAL FAMILY SPECIFIC INFORMATION ################
## To add a new journal family, add an entry to the journals 
## dictionary below. Fill out the fields according to the 'sample' 
## entry.
#####################################################################
journals = {
    "sample" :
        {'domain':      "The domain name of journal family",
        ## the next 6 fields determine the url of journals search page
         'relevant':    "The value for the journal's sorting field" +
                        "that donotes sorting search results based" + 
                        "on relevance in the search page url",
         'recent':      "The sorting field value for recency",
         'path':        "The portion of the url that runs from the" +
                        "end of the domain to search terms",
         'join':        "The portion of the url that appears between" +
                        "consecutive search terms",
         'pre_sb':      "Portion of the url that extends from the" +
                        "search terms to the paremeter value for" +
                        "sorting search results",
         'post_sb':     "The final portion of the url", 
         ## used for get_article_delimiters
         'article_path':"The journal's url path to articles",
        },
    "acs" :
        {'domain':      "https://pubs.acs.org",
         'relevant':    "relevancy",
         'recent':      "Earliest",
         'path':        "/action/doSearch?AllField=\"",
         'join':        "\"+\"",
         'pre_sb':      "\"&publication=&accessType=allContent&Earliest=&pageSize=20&startPage=0&sortBy=",
         'post_sb':     "",
         'article_path':('/doi/',['abs','full','pdf']),
        },
    "nature" :
        {'domain' :     "https://www.nature.com",
         'relevant':    "relevance",
         'recent':      "date_desc",
         'path':        "/search?q=\"",
         'join':        "\"%20\"",
         'pre_sb':      "\"&order=",
         'post_sb':     "&page=1",
         'article_path':('/articles/',''),
        },
    "rsc" :
        {'domain' :     "https://pubs.rsc.org",
         'relevant':    "Relevance",
         'recent':      "Latest%20to%20oldest",
         'path':        "/en/results?searchtext=",
         'join':        "\"%20\"",
         'pre_sb':      "\"&SortBy=",
         'post_sb':     "&PageSize=1&tab=all&fcategory=all&filter=all&Article%20Access=Open+Access",
         'article_path':('/en/content/articlehtml/',''),
         },
}

def get_soup_from_request(url: str, fast_load=True):
    """
    Get a BeautifulSoup parse tree (lxml parser) from a url request 

    Args:
        url: A requested url 
    Returns:
        A BeautifulSoup parse tree.
    """
    dynamic_list = ["https://pubs.rsc.org/"]

    if len([True for a in dynamic_list if a in url])>0:

        chrome_options = Options()  
        chrome_options.add_argument("--headless")
        chrome_options.binary_location=os.path.dirname(__file__)+"/journals/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary"
        driver = webdriver.Chrome(executable_path=os.path.dirname(__file__)+"/journals/chromedriver",chrome_options=chrome_options) 

        driver.get(url)
        if fast_load:
            time.sleep(1)
            soup = BeautifulSoup(driver.page_source, 'lxml')
        else:
            loaded_entries = BeautifulSoup(driver.page_source, 'lxml').text.count('https://doi.org/')
            while loaded_entries < np.min([1000,int(url.split('&PageSize=')[-1].split('&tab=all')[0])]):
                time.sleep(0.1)
                loaded_entries = BeautifulSoup(driver.page_source, 'lxml').text.count('https://doi.org/')
            # print("Entries fully loaded: {}".format(loaded_entries))
        soup = BeautifulSoup(driver.page_source, 'lxml')
        driver.quit()
    else:
        with requests.Session() as session:
            r = session.get(url) 
        soup =BeautifulSoup(r.text, 'lxml')
    return soup

def get_domain_name(search_query: dict) -> str:
    """
    Get url base path (domain name) for the requested journal family in search_query.

    Args:
        search_query: A query json
    Returns:
        The domain name of the journal, as a string
    """
    journal_family = search_query['journal_family'].lower()
    if journal_family not in journals:
        raise NameError('journal family {0} is not defined'.format(search_query['journal_family'].lower()))

    return journals[journal_family]['domain']

def get_search_parameters(search_query: dict) -> str:
    """
    Get prioritized url extensions (path and search parameters) for search terms in search_query.

    Args:
        search_query: A query json
    Returns:
        A list of url paths and queries as strings
    """
    journal_family = search_query['journal_family'].lower()
    if journal_family not in journals:
        raise NameError('journal family {0} is not defined'.format(journal_family))
    ## creates a list of search terms
    search_list = [[search_query['query'][key]['term']]+search_query['query'][key]['synonyms'] for key in search_query['query'] if len(search_query['query'][key]['synonyms'])>0]
    search_product = list(itertools.product(*search_list))
    extensions = []

    # save journal family's dictionar as variable
    jdict = journals[journal_family]
    # sortby is the requested method to sort results (relevancy or recency) and
    # sbtext gives the journal specific parameter to sort as requested
    sortby = search_query['sortby']
    sbext = journals[journal_family][sortby]  

    # this creates the url (excluding domain and protocol) for each search query
    for search_group in search_product:
        url_path_and_query = (jdict['path']
            + jdict['join'].join(["+".join(a.split(" ")) for a in search_group])
            + jdict['pre_sb'] + sbext + jdict['post_sb'])
        extensions.append(url_path_and_query)
    
    return extensions

def get_article_delimiters(search_query: dict) -> tuple:
    """
    Get delimiters specific to articles from requested journal family in search_query

    Args:
        search_query: A query json
    Returns:
        (delimiter for articles in the journal family url, \
        list containing delimiters for possible reader versions for an article).
    """
    journal_family = search_query['journal_family'].lower()
    if journal_family not in journals:
        raise NameError('journal family {0} is not defined'.format(journal_family))
    return journals[journal_family]['article_path']    

def get_page_info(search_query: dict) -> tuple:
    """
    Get index origin and total number of pages from search_query.

    Args:
        search_query: A query json
        response: A requests.Response (contains a server’s response to an HTTP request)
    Returns:
        (index origin, total page count in search)
    """

    request = create_request(search_query,-1)
    soup = get_soup_from_request(request)

    if search_query['journal_family'].lower() == "nature":
        parsed = str('{'+soup.text.split(',"keywords":')[0].split('"search":{')[-1].replace('"', "'")+'}')
        search_info = ast.literal_eval(parsed)
    elif search_query['journal_family'].lower() == "acs":
        parsed = [a.split("of")[-1].strip() for a in soup.text.split("Results:")[1].split("Follow")[0].split('-')]
        search_info = {"totalPages":math.ceil(float(parsed[1])/20),"page":int(parsed[0])-1,"totalResults":int(parsed[1])}
    else:
        raise NameError('journal family {0} is not defined'.format(journal_family.lower()))

    return (search_info["page"],search_info["totalPages"],search_info["totalResults"])


def get_page_info_advanced(request: str, search_query: dict) -> tuple:
    """
    Get index origin and total number of pages from search_query.

    Args:
        search_query: A query json
        response: A requests.Response (contains a server’s response to an HTTP request)
    Returns:
        (index origin, total page count in search)
    """
    # request = create_request(search_query,-1)
    soup = get_soup_from_request(request, fast_load=True)

    if search_query['journal_family'].lower() == "nature":
        ## Finds total results, page number, and total pages in article html
        ## Data exists as json inside script tag with 'data-test'='dataLayer' attr.
        data_layer = soup.find(attrs = {'data-test': 'dataLayer'})
        data_layer_string = str(data_layer.string)
        data_layer_json = "{" + data_layer_string.split("[{", 1)[1].split("}];", 1)[0] + "}"
        parsed = json.loads(data_layer_json)
        search_info = parsed["page"]["search"]

    elif search_query['journal_family'].lower() == "acs":
        parsed = [a.split("of")[-1].strip() for a in soup.text.split("Results:")[1].split("Follow")[0].split('-')]
        search_info = {"totalPages":math.ceil(float(parsed[1])/20)-1,"page":0,"totalResults":int(parsed[1])}

    elif search_query['journal_family'].lower() == "rsc":
        possible_entries = [a.strip("\n") for a in soup.text.split(" - Showing page 1 of")[0].split("Back to tab navigation")[-1].split(" ") if a.strip("\n").isdigit()]
        if len(possible_entries) == 1:
            search_info = {"totalResults":possible_entries[0],"page":1,"totalPages":1}
        else:
            search_info = {"totalResults":0,"page":1,"totalPages":1}
    else:
        raise NameError('journal family {0} is not defined'.format(search_query['journal_family'].lower()))

    return (search_info["page"],search_info["totalPages"],search_info["totalResults"])


def turn_page(search_query: dict, url: str, pg_num: str, pg_size: str) -> str:
    """
    Create url for a GET request based on the search_query.

    Args:
        search_query: A query json
        pg: page number to search on (-1 = starts at index origin for journal)
    Returns:
        A url.
    """
    # url = get_base_url(search_query)+get_search_extension(search_query) # (default) starts at index origin for journal 
    # if pg == -1:
    #     return url
    if search_query['journal_family'].lower() == "nature":
        return url.split('&page=')[0]+'&page='+str(pg_num)
    elif search_query['journal_family'].lower() == "acs":
        return url.split('&startPage=')[0]+'&startPage='+str(pg_num)+"&pageSize="+str(20)
    elif search_query['journal_family'].lower() == "rsc":
        return url.split('1&tab=all')[0]+str(pg_size)+'&tab=all&fcategory=all&filter=all&Article%20Access=Open+Access'


def create_page1_requests(search_query: dict) -> str:
    """
    Create url for a GET request based on the search_query.

    Args:
        search_query: A query json
        pg: page number to search on (-1 = starts at index origin for journal)
    Returns:
        A url.
    """
    extensions = get_search_parameters(search_query)
    requests_list = []
    for extension in extensions:
        url = get_domain_name(search_query)+extension # (default) starts at index origin for journal 
        if search_query['journal_family'].lower() == "nature":
            requests_list.append(url.split('&page=')[0]+'&page='+str(1))
        elif search_query['journal_family'].lower() == "acs":
            requests_list.append(url)
        elif search_query['journal_family'].lower() == "rsc":
            requests_list.append(url)
    return requests_list


def get_article_extensions(search_query: dict) -> list:
    """
    Create a list of article url extensions from search_query

    Args:
        search_query: A query json
    Returns:
        A list of articles url extension from search.
    """
    extensions = []
    article_delim, reader_delims = get_article_delimiters(search_query)
    start,stop,total = get_page_info(search_query)
    for pg in range(start,stop+1):
        request = create_request(search_query,pg)
        soup = get_soup_from_request(request)
        for tags in soup.find_all('a',href=True):
            if len(tags.attrs['href'].split(article_delim)) > 1 :
                extensions.append(tags.attrs['href'])
    # Remove duplicate extensions for same articles with different reader types in name
    extensions = [a for a in extensions if not \
                  len(set(reader_delims).intersection(set(a.split("/"))))>0]
    return extensions[0:search_query["maximum_scraped"]]


def get_article_extensions_advanced(search_query: dict) -> list:
    """
    Create a list of article url extensions from search_query

    Args:
        search_query: A query json
    Returns:
        A list of articles url extension from search.
    """
    extensions = []
    article_delim, reader_delims = get_article_delimiters(search_query)
    page1_requests = create_page1_requests(search_query)
    articles_found = 0
    for page1 in page1_requests:
        print("GET request: ",page1)
        page_returns = []
        start,stop,total = get_page_info_advanced(page1,search_query)
        for pg_num in range(start,stop+1):
            request = turn_page(search_query,page1,pg_num,total)
            soup = get_soup_from_request(request,fast_load=False)
            for tags in soup.find_all('a',href=True):
                if len(tags.attrs['href'].split(article_delim)) > 1 :
                    page_returns.append(tags.attrs['href'])
        extensions.append(page_returns)
        
        #check if we have found enough articles
        articles_found += len(page_returns)
        if articles_found > search_query["maximum_scraped"]:
            break
    
    extensions = list(itertools.chain(*itertools.zip_longest(*extensions)))
    extensions = [a for a in extensions if a != None]
    extensions = [a for a in extensions if not \
                  len(set(reader_delims).intersection(set(a.split("/"))))>0]
    extensions = [a.split('?page=search')[0] for a in extensions]
   
    extensions = list(OrderedDict.fromkeys(extensions))
    return extensions[0:search_query["maximum_scraped"]]

def get_article_figures(url: str, save_path="") -> dict:
    """
    Get all figures from an article 

    Args:
        url: A url to a journal article
        save_path: A location to save extracted figures
    Returns:
        A dict of figure_jsons from an article
    """
    soup = get_soup_from_request(url)

    # Uncomment to save html
    os.makedirs(save_path+ "/html/", exist_ok=True)
    with open(save_path+ "/html/" + url.split("/")[-1]+'.html', "w", encoding='utf-8') as file:
        file.write(str(soup))

    if 'acs' in url.split("."):
        prepend = "https://pubs.acs.org"
        extra_key = "inline-fig internalNav"
        figure_list = [a for a in soup.find_all('figure') if str(a).find(extra_key)>-1]
    elif 'nature' in url.split("."):
        prepend = ""
        extra_key = " "
        figure_list = [a for a in soup.find_all('figure') if str(a).find(extra_key)>-1]
    elif 'rsc' in url.split("."):
        prepend = "https://pubs.rsc.org"
        extra_key = "/image/article"
        figure_list = soup.find_all("div", class_="image_table")
    else:
        raise NameError('url destination not recognized')

    figures = 1
    article_json = {}

    # for figure in soup.find_all('figure'):
    for figure in figure_list:
        
        if 'rsc' in url.split("."):
            captions = figure.find_all("span", class_="graphic_title")
        else:
            captions = figure.find_all('p')

        # print("Caption: ",captions)

        # acs captions are duplicated, one version with no captions
        if len(captions) == 0:
            continue
        
        # initialize the figure's json
        article_name = url.split("/")[-1]
        figure_json = {"title": soup.find('title').get_text(), "article_url" : url, "article_name" : article_name}
        
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
            for image_tag in [a for a in figure.find_all("a", href=True) if str(a).find(extra_key)>-1]:
                image_url = image_tag['href']
        else:
            image_tag = figure.find('img')
            image_url = image_tag.get('src')

        image_url = prepend + image_url.replace('_hi-res','')
        if ":" not in image_url:
            image_url = "https:" + image_url
        figure_name = article_name + "_fig" + str(figures) + ".jpg"  #" +  image_url.split('.')[-1]

        # save image info
        figure_json["figure_name"] = figure_name
        figure_json["image_url"] = image_url

        # save figure as image
        if save_path:
            os.makedirs(save_path+ "/figures/", exist_ok=True)

            if 'rsc' in url.split("."):
                out_file = save_path + "/figures/" + figure_name
                urllib.request.urlretrieve(image_url, out_file)
                figure_json["figure_path"] = save_path + "figures/" + figure_name                             
            else:
                response = requests.get(image_url, stream=True)
                with open(save_path + "/figures/" + figure_name, 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)
                del response 
                figure_json["figure_path"] = save_path + "figures/" + figure_name

        else:
            figure_json["figure_path"] = "" 

        figure_json["master_images"] = []
        figure_json["unassigned"] = {'master_images':[],'dependent_images':[],'inset_images':[],\
                                     'subfigure_labels':[],'scale_bar_labels':[],\
                                     'scale_bar_lines':[],'captions':[]}

        # add all results
        article_json[figure_name] = figure_json
        
        # increment index
        figures += 1

    return article_json