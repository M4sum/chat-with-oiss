# # -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()

from bs4 import BeautifulSoup as Soup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from urllib.parse import urljoin, urlparse
import os
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_documents(url):
    print(f"Fetching content from {url}")
    visited_urls = set()
    documents = []
    session = requests.Session()
    
    def extract_content(soup):
        content_div = soup.find('div', {'class': 'content'})
        if content_div:
            documents.append(content_div.text.strip())
        if len(visited_urls)%100 == 0:
            print(f"Visited urls: {len(visited_urls)}")

    def extract_links(soup, base_url):
        # print(visited_urls, len(documents))
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not href.startswith(('http', 'mailto', '#', 'index.html')):
                url = urljoin(base_url, href)
                if url not in visited_urls:
                    visited_urls.add(url)
                    process_url(url)
        print(f"Visited urls: {len(visited_urls)}")

    def process_url(url):
        try:
            response = session.get(url)
            if response.status_code == 200:
                soup = Soup(response.content, 'html.parser')
                extract_content(soup)
                extract_links(soup, url)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while processing URL: {url}")
            print(e)

    process_url(url)
    text = "\n\n".join(documents)
    return text


def url_exists(url):
    response = requests.head(url)
    return response.status_code == 200

def save_text_chunks(text_chunks, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        for chunk in text_chunks:
            file.write(chunk + '\n')