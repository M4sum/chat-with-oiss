from bs4 import BeautifulSoup as Soup
from cachetools import TTLCache
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdb
import requests
from requests import Session
from urllib.parse import urljoin, urlparse

session = Session()
cache = TTLCache(maxsize=1000, ttl=3600)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_documents(url):
    visited_urls = set()
    documents = []

    session = requests.Session()

    def extract_content(soup):
        content_div = soup.find('div', {'class': 'content'})
        if content_div:
            documents.append(content_div.text.strip())

    def extract_links(soup, base_url):
        # pdb.set_trace()
        # print(visited_urls, len(documents))
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not href.startswith(('http', 'mailto', '#', 'index.html')):
                url = urljoin(base_url, href)
                if url not in visited_urls:
                    visited_urls.add(url)
                    process_url(url)

    def process_url(url):
        try:
            response = session.get(url)
            if response.status_code == 200:
                soup = Soup(response.content, 'html.parser')
                # pdb.set_trace()
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

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
