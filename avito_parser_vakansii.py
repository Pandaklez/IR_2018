# python3


import requests
from bs4 import BeautifulSoup
from random import randint
from time import sleep
import re
import json


def jobs_page_get(page_num):
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'
    # хотим притворяться браузером
    req = requests.get('https://www.avito.ru/rossiya/vakansii/prodazhi' + '?p=' + str(page_num),
                       headers={'User-agent': user_agent})
    b_soup = BeautifulSoup(req.content, 'html5lib', from_encoding=req.encoding)
    return b_soup


def get_page_urls(soup):
    '''
    :param soup: beautiful soup from jobs_page_get() function
    :return: list of strings (of pages' urls)
    length of urls var is 10 000
    '''
    links = soup.find_all("a", class_="item-description-title-link")
    href_list = []
    for link in links[:10000]:
        href = "https://www.avito.ru" + link.get('href')
        href_list.append(href)
    print(len(href_list))
    return href_list


def get_text(page):
    '''
    :param page: is a soup of the page
    :return: page text as a string
    '''

    page_text = page.find_all("div", class_="item-description")[0].get_text()
    return page_text


def get_metadata(page):
    '''
    :param page: is a soup of the page
    :return: dictionary with the metadata on the job
    meta = {}
    '''
    posted_at = page.find_all("div", class_="title-info-metadata-item")[0].get_text().strip("\n  ")
    try:
        company_address = page.find_all("div", class_="item-map-location")[0].get_text().strip("\n  ")
    except IndexError:
        company_address = 'None'
    job_title = page.find_all("div", class_="sticky-header-prop sticky-header-title")[0].get_text().strip("\n  ")
    try:
        salary = page.find_all("div", class_="price-value price-value_side-card")[0].get_text().strip("\xa0₽\n  ")
    except IndexError:
        salary = page.find_all("div", class_="price-value price-value_side-card price-value_size-small")[0].get_text().strip("\xa0₽\n  ")
    employer = page.find_all("div", class_="seller-info-name")[0].get_text().strip("\n  ")

    three_params = page.find_all("div", class_="item-params item-params_type-one-colon")[0].get_text()

    schedule = re.search('к работы: (.*)', three_params)
    exp = re.search('т работы: (.*)', three_params)
    if schedule:
        job_type = schedule.group(1)
    else:
        job_type = "?"
    if exp:
        experience = exp.group(1)
    else:
        experience = "?"

    meta = {"posted_at": posted_at, "company_address": company_address, "job_title": job_title,
            "salary": salary, "employer": employer, "job_type": job_type, "experience": experience}

    return meta


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', string)


def write_in_file(text, meta, url):
    # each file has to be a json inside
    '''
    :param text: string
    :param meta: dict
    as a result write this into jason file
    name of the file is url of the page
    '''
    d = {"text": remove_emoji(text), "meta": meta}
    with open('avito\\' + url[-16:] + '.json', 'w') as outfile:
        json.dump(d, outfile, ensure_ascii=False)


if __name__ == '__main__':
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'

    # this chunk is for getting urls_list
    '''
    urls = []
    for page_num in range(1, 50):
        bs = jobs_page_get(page_num)
        urls.extend(get_page_urls(bs))
        sleep(randint(1, 10)/10.0)

    print(len(urls))  # 5339 for now, 5342 second time
    with open('urls.txt', 'w') as the_file:
        the_file.write(str(urls))
    '''

    '''
    url_address = 'https://www.avito.ru/moskovskaya_oblast_krasnogorsk/vakansii/prodavets-konsultant_1690337918'
    r = requests.get(url_address,
                     headers={'User-agent': user_agent})
    soup = BeautifulSoup(r.content, 'html5lib', from_encoding=r.encoding)
    text = get_text(soup)
    metadata = get_metadata(soup)
    write_in_file(text, metadata, url_address)
    '''

    # make urls_list out of urls.txt file
    with open('urls.txt', 'r') as f:
        output = f.read()
    output = output.strip('[\'\']')
    output = output.split('\', \'')
    urls_list = output

    for url_address in urls_list[411:2510]: # 4544
        print(url_address)
        r = requests.get(url_address,
                         headers={'User-agent': user_agent})
        soup = BeautifulSoup(r.content, 'html5lib', from_encoding=r.encoding)
        try:
            text = get_text(soup)
        except IndexError:
            continue
        try:
            metadata = get_metadata(soup)
        except IndexError:
            continue
        try:
            write_in_file(text, metadata, url_address)
        except UnicodeEncodeError:
            continue
        sleep(randint(1, 90)/10.0)
