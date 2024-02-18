import newspaper
from newspaper import Article
from newspaper import Source
import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from ua_parser import user_agent_parser
from functools import cached_property
from typing import List
import time
import random
import httpx
from user_agent import Rotator


def scrap_one_article(link):
    nltk.download('punkt')

    article = Article(link)
    article.download()
    article.parse()
    article.nlp()

    # To print out the full text
    print(article.text)

    # To print out a summary of the text
    # This works, because newspaper3k has built in NLP tools
    print(article.summary)

    # To print out the list of authors
    print(article.authors)

    # To print out the list of keywords
    print(article.keywords)

    # Other functions to gather the other useful bits of meta data in an article
    article.title  # Gives the title
    article.publish_date  # Gives the date the article was published
    article.top_image  # Gives the link to the main image of the article
    article.images  # Provides a set of image links


def scrap_multiple_articles(base_link, db_name):
    nltk.download('punkt')
    gamespot = newspaper.build(base_link,
                               memoize_articles=False)
    # I set memoize_articles to False, because I don't want it to cache and save the articles to memory, run after run.
    # Fresh run, everytime we run execute this script essentially
    final_df = pd.DataFrame()
    i = 0
    for each_article in gamespot.articles:
        print(++i)
        try:
            each_article.download()
            each_article.parse()
            each_article.nlp()

        except:
            continue

        temp_df = pd.DataFrame(columns=['Title', 'Authors', 'Text',
                                        'Summary', 'published_date', 'Source'])

        temp_df['Authors'] = each_article.authors
        temp_df['Title'] = each_article.title
        temp_df['Text'] = each_article.text
        temp_df['Summary'] = each_article.summary
        temp_df['published_date'] = each_article.publish_date
        temp_df['Source'] = each_article.source_url

        final_df = final_df.append(temp_df, ignore_index=True)

    # From here you can export this Pandas DataFrame to a csv file
    final_df.to_csv(db_name + '.csv', encoding="utf-8")


def section_loop_scrap(db_name):
    root = 'https://polytech.univ-nantes.fr/'
    articles_section = f'{root}/fr'
    website = 'https://polytech.univ-nantes.fr/fr/les-formations/cycle-ingenieur'
    result = requests.get(website)
    soup = BeautifulSoup(result.text, 'lxml')
    left_navigation = soup.find('div', id='navigation')
    links = [link['href'] for link in left_navigation.find_all('a', href=True)]
    print(links)
    links_2 = []

    for link in links:
        if link.find(root) != -1:
            print('wops')
            result = requests.get(link)
            content = result.text
            soup = BeautifulSoup(content, 'lxml')
            left_navigation = soup.find('div', id='navigation')
            submenus = left_navigation.find_all('li')
            for submenu in submenus:

                sub_elements = submenu.find_all('ul')
                for menu in sub_elements:
                    sublinks = [link['href'] for link in menu.find_all('a', href=True)]
                    for sublink in sublinks:
                        if sublink not in links_2:
                            links_2.append(sublink)
    for link in links_2:
        if link not in links:
            links.append(link)

    print(links)
    count = 0
    data = pd.DataFrame(columns=['id', 'link', 'content'])
    for link in links:
        if link.find(root) != -1:
            print(count)
            result = requests.get(link)
            content = result.text
            soup = BeautifulSoup(content, 'lxml')
            # title = soup.find('div', id='zone-titre').find('h1').get_text()
            text_content = soup.find('div', id='contenu').get_text()
            temp_data = {
                'id': [count],
                'link': [link],
                'content': [text_content]
            }
            temp_data = pd.DataFrame(temp_data, columns=['id', 'link', 'content'])
            # temp_data['id'] = count
            # temp_data['link'] = link
            # temp_data['content'] = text_content
            print(temp_data)
            data = data.append(temp_data)
            count = count + 1

    data.to_csv(db_name + '.csv', encoding="utf-8")

    ####
    # result = requests.get(website)
    # content = result.text
    # soup = BeautifulSoup(content, 'lxml')
    #
    # title = soup.find('div', id='zone-titre').find('h1').get_text()
    # content = soup.find('div', id='contenu').get_text()
    #


def find_current_page(page):
    page_counter = page.find('span', {"class": "search-pagination__current-page"})
    counters = []
    for s in page_counter.split():
        if s.isdigit():
            counters.append(int(s))
    return counters[0]


def find_max_page(page):
    page_counter = page.find('span', {"class": "search-pagination__current-page"})
    counters = []
    for s in page_counter.get_text().split():
        if s.isdigit():
            counters.append(int(s))
    return counters[1]


def scrap_with_search(tokens, max_pages_result, max_results, load_from_file):
    rotator = Rotator(find_user_agent_list())
    root = 'univ-nantes.fr'
    search_root = 'https://polytech.univ-nantes.fr/search?q='
    links = []
    # Links collection Part
    if not load_from_file:
        for token in tokens:
            print("current token: " + token)
            page = 1
            website = f'{search_root}{token}&limit=10&page={page}'
            try:
                result = requests.get(website)
            except:
                time.sleep(60)
                print("An exception occurred")
                continue

            soup = BeautifulSoup(result.text, 'lxml')
            max_pages = find_max_page(soup)
            for i in range(1, min(max_pages, max_pages_result)):
                print("current page: " + str(i) + "/" + str(min(max_pages, max_pages_result)))
                current_website = f'{search_root}{token}&limit=10&page={i}'
                # time.sleep(1.5)
                try:
                    result = requests.get(current_website, headers={"User-Agent": rotator.get().string})
                except:
                    time.sleep(60)
                    print("An exception occurred")
                    continue
                soup = BeautifulSoup(result.text, 'lxml')
                for link in soup.find_all('a', attrs={"href": True, "class": "item-title__element_title"}):
                    if link not in links:
                        links.append(link['href'])

        print("finished extracting links")
        try:
            with open(r'/home/pred_index_23/scrap-data/links', 'w') as fp:
                for item in links:
                    # write each item on a new line
                    fp.write("%s\n" % item)
                print('Done writing links')
        except:
            print('links not saved')
    else:
        links = load_list_from_file()

    # Content scrapping part

    missed_links = []
    count = 0
    data = pd.DataFrame(columns=['id', 'link', 'content'])
    for link in links:
        if link.find(root) != -1:
            print("extracting ( " + str(count) + " / " + str(len(links)) + ") : " + link)
            # time.sleep(1.5)
            try:
                result = requests.get(link, headers={"User-Agent": rotator.get().string})
            except:
                time.sleep(60)
                print("An exception occurred")
                missed_links.append(link)
                continue
            content = result.text
            soup = BeautifulSoup(content, 'lxml')
            # title = soup.find('div', id='zone-titre').find('h1').get_text()
            text_content_div = soup.find('div', id='contenu')
            # text_content = soup.find('div', id='contenu').get_text()
            if text_content_div is None:
                continue
            text_content = text_content_div.get_text()
            temp_data = {
                'id': [count],
                'link': [link],
                'content': [text_content]
            }
            temp_data = pd.DataFrame(temp_data, columns=['id', 'link', 'content'])
            # temp_data['id'] = count
            # temp_data['link'] = link
            # temp_data['content'] = text_content
            # data = data.concat(temp_data)
            data = pd.concat([data, temp_data])
            count = count + 1

    data.to_csv('/home/pred_index_23/scrap-data/scrapv4.csv', encoding="utf-8", index=False)
    try:
        with open(r'/home/pred_index_23/scrap-data/missed_links', 'w') as fp:
            for item in missed_links:
                # write each item on a new line
                fp.write("%s\n" % item)
            print('Done writing links')
    except:
        print('links not saved')
    # data.to_csv('scrapv3.csv', encoding="utf-8")


def find_user_agent_list():
    url = "https://www.useragentlist.net/"
    request = httpx.get(url)
    user_agents = []
    soup = BeautifulSoup(request.text, "html.parser")
    for user_agent in soup.select("pre.wp-block-code"):
        user_agents.append(user_agent.text)
    return user_agents


def load_list_from_file():
    links = []
    infile = open('/home/pred_index_23/scrap-data/links', 'r')
    for line in infile:
        links.append(line.replace("\n", ""))

    infile.close()
    return links


if __name__ == '__main__':
    tokens = ["Polytech Nantes site officiel",
              "Programmes d'ingénierie Polytech Nantes",
              "Admissions Polytech Nantes",
              "Cours et filières Polytech Nantes",
              "Stages Polytech Nantes",
              "Vie étudiante Polytech Nantes",
              "Associations étudiantes Polytech Nantes",
              "Projets de recherche Polytech Nantes",
              "Partenariats industriels Polytech Nantes",
              "Événements et actualités Polytech Nantes",
              "Étudiants internationaux Polytech Nantes",
              "Laboratoires de recherche Polytech Nantes",
              "Alumni Polytech Nantes",
              "Enseignants Polytech Nantes",
              "Bourses d'études Polytech Nantes",
              "Installations sportives Polytech Nantes",
              "Cafétéria Polytech Nantes",
              "Bibliothèque Polytech Nantes",
              "Échange étudiant Polytech Nantes",
              "Double diplôme Polytech Nantes"]
    rotator = Rotator(find_user_agent_list())

    scrap_with_search(tokens, 1000, 0, False)
    # scrap_multiple_articles(https://polytech.univ-nantes.fr/fr/une-ecole-sur-3-campus/actualites", "context_scrap")
    # section_loop_scrap("scrapv2")
