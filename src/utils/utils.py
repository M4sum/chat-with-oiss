from bs4 import BeautifulSoup as Soup
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdb
import requests

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def fetch_text_from_url(url):
    response = requests.get(url)

    soup = Soup(response.text, 'html.parser')

    content_divs = soup.find_all('div', class_='content')

    def remove_text_by_id(tag, target_id):
        target_tag = tag.find(id=target_id)
        if target_tag:
            target_tag.clear()


    for div in content_divs:
        remove_text_by_id(div, "breadcrumbs")

    extracted_content = []
    for div in content_divs:
        # content_text = div.get_text(strip=True)
        # tags_to_extract = [tag for tag in div.find_all(True)]
        
        # extracted_content.append('\n'.join(tag.get_text() for tag in tags_to_extract))
        extracted_content.append(div.get_text(strip=True))
        # print(extracted_content)
    return extracted_content

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