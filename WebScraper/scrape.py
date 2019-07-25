import os
import time
import json
import errno
import random
import requests
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
    rp = random.randint(0,8)+random.randint(1,10)/10.0+random.randint(1,100)/100.0
    daily_routines = ["Do the laundry","Hang the clothes","Iron the clothes","Make the bed","Go to bed","Wake up","Brush the teeth","Drive to work","Get home","Take a bath","Brush your hair","Surf the net","Play with friends","Go to school","Go shopping","Exercise","Wash the car","Get dressed","Go out with a friend","Take pictures","Play the guitar","Water the plant","Go for a walk","Work","Have breakfast","Have lunch","Have dinner","Make dinner","Fold the laundry","Surf the net","Feed the dog","Take a taxi","Wait for the bus","Paint the picture","Take a break","Walk the dog","Take out the rubbish","Sweep the floor","Rake the leaves","Read the news","Clean the window","Cut the grass","Do the dishes","Paint the house"]
    print("I will "+random.choice(daily_routines).lower()+" for "+str(np.round(rp,3))+" seconds.\n")
    time.sleep(rp)

def get_request_base_url(journal):
    if journal.lower() == "nature":
        base = "https://www-nature-com"
    elif journal.lower() == "acs":
        base = "https://pubs-acs-org"
    else:
        raise NameError('journal {0} is not defined'.format(journal.lower()))
    return base

def get_search_extension(journal,search_list):
    if journal.lower() == "nature":
        return '/search?'+"q="+",%20".join(["+".join(a.split(" ")) for a in search_list])+"&order=relevance&page=1"
    elif journal.lower() == "acs":
        return '/action/doSearch?AllField='+'%2C'.join(["+".join(a.split(" ")) for a in search_list])+'&startPage=0&pageSize=20'
    else:
        raise NameError('journal {0} is not defined'.format(journal.lower()))

def get_directory_parser(journal):
    if journal.lower() == "nature":
        return ('/articles/','')
    elif journal.lower() == "acs":
        return ('/doi/',['abs','full','pdf'])

def get_page_count(journal,soup_text):
    if journal.lower() == "nature":
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

def create_payload(dict_json):
    return {dict_json['payload']['token1']['name']:dict_json['payload']['token1']['value'],\
            dict_json['payload']['token2']['name']:dict_json['payload']['token2']['value']}

def create_request(dict_json):
    # Parses input json into formal request (for python requests package) 
    request_base_url = get_request_base_url(dict_json['journal_family'])
    search_extension = get_search_extension(dict_json['journal_family'],dict_json['query'])
    
    try:
        os.makedirs(dict_json["output_scraped"])
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    return [dict_json['post_login_url'],request_base_url+dict_json['inst_extension'],search_extension]

def main():
    """
    Scrape html files from specified journals by keyword
    """
    dict_json = js_r("key.json")

    payload = create_payload(dict_json)
    request = create_request(dict_json) #[POST_URL,GET_URL_BASE,SEARCH_EXTENSION]

    directory_parser = get_directory_parser(dict_json['journal_family'])
    
    # Session #1: Get article url extensions for articles related to query
    with requests.Session() as session:
        post = session.post(request[0],data=payload)
        get  = session.get(request[1]+request[2])
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
            request_pg = get_page_extension(dict_json['journal_family'],request[1]+request[2],pg_num)
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
        post = session.post(request[0],data=payload)
        for article in article_extensions:
            print("["+str(counts+1).zfill(5)+"] Extracting html from "+str(article.split("/")[-1]))       
            r = session.get(request[1]+article) 
            with open(dict_json['output_scraped']+"/"+article.split("/")[-1]+".html", 'wb') as f:
                f.write(r.content)
            if dict_json['human_traffic']:
                human_traffic()
            if counts+1 == dict_json['maximum_scraped']:
                break
            counts+=1

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("\nTime Elapsed = ",time.time()-start_time)