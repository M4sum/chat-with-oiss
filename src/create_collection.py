from crud_collections import *
from data.make_dataset import get_documents, get_text_chunks, save_text_chunks
from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os
from qdrant_client import QdrantClient

load_dotenv()
url = "https://www.northwestern.edu/international/international-students/"
host_url = os.getenv("QDRANT_HOST")
api_key=os.getenv("QDRANT_API_KEY")
collection_name="Northwestern University"
embedding_dim = 384 #fastembed; 
# embedding_dim = 1536 #openai
data_path = "data/text_chunks.txt"

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = FastEmbedEmbeddings()
client = QdrantClient(url=host_url, api_key=api_key)
vectorstore = create_qdrant_collection(client, collection_name, embeddings, embedding_dim)

# raw_text = get_documents(url)
# text_chunks = get_text_chunks(raw_text)
# save_text_chunks(text_chunks, data_path)

with open(data_path, 'r') as file:
    text_chunks = file.readlines()
vectorstore = get_qdrant_vectorstore(client, embeddings, collection_name)
vectorstore.add_texts(text_chunks)