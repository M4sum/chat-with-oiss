from bs4 import BeautifulSoup as Soup
from cachetools import TTLCache
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_anthropic import ChatAnthropic
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pdb
import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.http import models
import requests
from requests import Session
import streamlit as st
from urllib.parse import urljoin, urlparse

session = Session()
cache = TTLCache(maxsize=1000, ttl=3600)
template = """You are an expert chatbot, tasked with answering any 
immigration question from the Northwestern OISS Wbesite.

Generate a comprehensive and informative answer of 100 words or less for the given question 
based solely on the provided search results (URL and content). You must only use information 
from the provided search results. Use an unbiased and journalistic tone. Combine search results
together into a coherent answer. Do not repeat text.

You should use bullet points in your answer for readability. You should highlight the passage you
fetched yor answer from with this notation: $relevant passage$.

If there is nothing in the context relevant to the question at hand, just say "Hmm.. I don't know" 
Don't try to make up an answer.
"""

prompt=PromptTemplate.from_template(
    template=template
)

# prompt = ChatPromptTemplate(
#     messages=[
#         SystemMessagePromptTemplate.from_template(
#             """You are an expert programmer and problem-solver, tasked with answering any 
#             immigration question from the Northwestern OISS Wbesite.

#             Generate a comprehensive and informative answer of 100 words or less for the given question 
#             based solely on the provided search results (URL and content). You must only use information 
#             from the provided search results. Use an unbiased and journalistic tone. Combine search results
#             together into a coherent answer. Do not repeat text.

#             You should use bullet points in your answer for readability. You should highlight the passage you
#             fetched yor answer from with this notation: $relevant passage$.

#             If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." 
#             Don't try to make up an answer."""
#         ),
#         # The `variable_name` here is what must align with memory
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{question}")
#     ]
# )

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

def get_conversation_chain(vectorstore, msgs=None):
    
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', chat_memory=msgs, return_messages=True)
    llm = ChatAnthropic(temperature=0, \
                        model_name="claude-3-haiku-20240307")



    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        # condense_question_prompt=prompt,
        memory=memory,
        # return_source_documents=True
    )
    # pdb.set_trace()
    return conversation_chain

def create_qdrant_collection(client, collection_name):
    embeddings = OpenAIEmbeddings()

    client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=1536, 
        distance=models.Distance.COSINE),
)
    vector_store = Qdrant(
    client=client, collection_name=collection_name, 
    embeddings=embeddings,
)
    return vector_store

def get_qdrant_vectorstore(client, collection_name):
    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant(
    client=client, collection_name=collection_name, 
    embeddings=embeddings,
)
    return vector_store

def get_qdrant_collections(client):
    collection_names = []
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
    except Exception as e:
        print(f"Error--------------------\n\n{e}\n\n\n")
        st.error("Uh-oh! the Qdrant cluster is inactive. Free qdrant clusters become inactive after few days of inactivity.\
                 Please contact me through my email or Linkedin and I'll try to get it up and running as soon as I can. Thank you and apologies for the inconvenience :)")
    return collection_names