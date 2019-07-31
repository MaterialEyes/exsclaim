import os
import time
import json
import errno
import unicodedata
import random

import requests
import shutil
import numpy as np
from bs4 import BeautifulSoup

def js_r(filename):
    with open(filename) as f_in:
        return(json.load(f_in))

def exist_common_member(a, b): 
    if len(set(a).intersection(set(b))) > 0: 
        return(True)  
    return(False)

def human_traffic():
    rp = random.randint(0,12)+random.randint(1,10)/10.0+random.randint(1,100)/100.0
    daily_routines = ["Do the laundry","Hang the clothes","Iron the clothes","Make the bed","Go to bed","Wake up","Brush the teeth","Drive to work","Get home","Take a bath","Brush your hair","Surf the net","Play with friends","Go to school","Go shopping","Exercise","Wash the car","Get dressed","Go out with a friend","Take pictures","Play the guitar","Water the plant","Go for a walk","Work","Have breakfast","Have lunch","Have dinner","Make dinner","Fold the laundry","Surf the net","Feed the dog","Take a taxi","Wait for the bus","Paint the picture","Take a break","Walk the dog","Take out the rubbish","Sweep the floor","Rake the leaves","Read the news","Clean the window","Cut the grass","Do the dishes","Paint the house"]
    print("I will "+random.choice(daily_routines).lower()+" for "+str(np.round(rp,3))+" seconds.\n")
    time.sleep(rp)

def get_request_base_url(journal):
    if journal.lower() == "nature":
        base = "https://www.nature.com"
    elif journal.lower() == "acs":
        base = "https://pubs.acs.org"
    else:
        raise NameError('journal {0} is not defined'.format(journal.lower()))
    return base

def get_search_extension(journal,search_list,sortby):
    if journal.lower() == "nature":
        if sortby == "recent":
            sbext = "&order=date_desc"
        else:
            sbext = "&order=relevance"
        return '/search?'+"q="+",%20".join(["+".join(a.split(" ")) for a in search_list])+sbext+"&page=1"
    elif journal.lower() == "acs":
        if sortby == "recent":
            sbext = "&sortBy=Earliest"
        else:
            sbext = "&sortBy=relevancy"
        return '/action/doSearch?'+"".join(["&field"+str(i+1)+"=AllField&text"+str(i+1)+"="+"+".join(search_list[i].split(" ")) for i in range(len(search_list))])+"&publication=&accessType=allContent&Earliest=&pageSize=20&startPage=0"+sbext
    else:
        raise NameError('journal {0} is not defined'.format(journal.lower()))

def get_directory_parser(journal):
    if journal.lower() == "nature":
        return ('/articles/','')
    elif journal.lower() == "acs":
        return ('/doi/',['abs','full','pdf'])

def get_page_count(journal,soup_text):
    if journal.lower() == "nature":
        #print(soup_text)
        return (1,int(soup_text.split("Next page")[0].split("page")[-1].strip()))
    elif journal.lower() == "acs":
        return (0,int(np.floor(int(soup_text.split("Results: 1 - 20of")[-1].split("Follow")[0])/20)))
    else:
        raise NameError('journal {0} is not defined'.format(journal.lower()))

def get_page_extension(journal,url,pg):
    if journal.lower() == "nature":
        return url.split('&page=')[0]+'&page='+str(pg)
    elif journal.lower() == "acs":
        return url.split('&startPage=')[0]+'&startPage='+str(pg)+"&pageSize=20"

def create_request(dict_json):
    # Parses input json into formal request (for python requests package) 
    request_base_url = get_request_base_url(dict_json['journal_family'])
    search_list = [dict_json['query'][key]['term'] for key in dict_json['query'] if len(dict_json['query'][key]['term'])>0]
    search_extension = get_search_extension(dict_json['journal_family'],search_list,dict_json['sortby'])
    return [request_base_url,search_extension]
    
def get_figures(soup, article_url, journal_family):
    exsclaim_json = {}
    title = soup.find('title').get_text()
    
    if journal_family == 'acs':
        prepend = "https://pubs.acs.org"
    elif journal_family == 'nature':
        prepend = ""
    
    figures = 1
    for figure in soup.find_all('figure'):
        captions = figure.find_all('p')
        
        # acs captions are duplicated, one version with no captions
        if len(captions) == 0:
            continue
        
        # initialize the figure's json
        article_name = article_url.split("/")[-1]
        figure_json = {"title": title, "article_url" : article_url, "article_name" : article_name}
        
        # get figure caption
        figure_caption = ""
        for caption in captions:
            figure_caption += caption.get_text()
        figure_json["caption"] = figure_caption
        
        # get figure url and name
        image_tag = figure.find('img')
        image_url = image_tag.get('src')
        image_url = prepend + image_url
        if ":" not in image_url:
            image_url = "https:" + image_url
        figure_name = article_name + "_fig" + str(figures) + "." + image_url.split('.')[-1]
        
        # save figure as image
        os.makedirs("images", exist_ok=True)
        response = requests.get(image_url, stream=True)
        with open("images/" + figure_name, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response 
        
        # save image info
        figure_json["figure_name"] : figure_name
        figure_json["image_url"] = image_url

        # add all results
        exsclaim_json[figure_name] = figure_json
        
        # increment index
        figures += 1
    
    return exsclaim_json

def main():
    """
    Scrape html files from specified journals by keyword
    """
    dict_json = js_r("query.json")

    request = create_request(dict_json) #[GET_URL_BASE,SEARCH_EXTENSION]

    directory_parser = get_directory_parser(dict_json['journal_family'])
    
    # Session #1: Get article url extensions for articles related to query
    with requests.Session() as session:
        #post = session.post(request[0],data=payload)
        get  = session.get(request[0]+request[1])
        soup = BeautifulSoup(get.text, 'lxml')
        start,total = get_page_count(dict_json['journal_family'],soup.text)
        
        print(""" 
             ____ ____ ____ ____ ____ ____ ____ ____ 
            ||e |||x |||s |||c |||l |||a |||i |||m ||
            ||__|||__|||__|||__|||__|||__|||__|||__||
            |/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|
            ||s |||c |||r |||a |||p |||i |||n |||g ||
            ||__|||__|||__|||__|||__|||__|||__|||__||
            |/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|            
        """)
        print("Retrieving relevant article paths from {0} pages...\n".format(total))
        
        article_extensions = []
        for pg_num in range(start,total+1):
            request_pg = get_page_extension(dict_json['journal_family'],request[0]+request[1],pg_num)
            r = session.get(request_pg)
            soup = BeautifulSoup(r.text, 'lxml')
            tags = soup.find_all('a',href=True)
            for t in tags:
                if len(t.attrs['href'].split(directory_parser[0])) > 1 :
                    article_extensions.append(t.attrs['href'])

        article_extensions = [a for a in article_extensions if not exist_common_member(directory_parser[1],a.split("/"))]

    # Session #2: Request and save html files from article url extensions
    with requests.Session() as session:
        counts=0
        exsclaim_json = {}
        for article in article_extensions:
            article_url = request[0] + article
            print("["+str(counts+1).zfill(5)+"] Extracting html from "+str(article.split("/")[-1]))       
            r = session.get(article_url) 
            soup = BeautifulSoup(r.text, 'lxml')
            article_figure_json = get_figures(soup, article_url, dict_json['journal_family'])
            
            exsclaim_json.update(article_figure_json)
            
            
            if dict_json['human_traffic']:
                human_traffic()
            if counts+1 == dict_json['maximum_scraped']:
                break
            counts+=1
    
    with open("exsclaim.json", "w") as f:
        json.dump(exsclaim_json, f)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("\nTime Elapsed = ",time.time()-start_time)
