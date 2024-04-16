from langchain_community.vectorstores import Qdrant
from qdrant_client.http import models
import streamlit as st

def create_qdrant_collection(client, collection_name, embeddings, embedding_dim, new_collection_name="", mode="recreate"):
    embeddings = embeddings

    if mode == "recreate":
        client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedding_dim,
            distance=models.Distance.COSINE),
    )
    elif mode == "from_existing":
        if not new_collection_name:
            raise Exception("specify new collection name if making collection from existing collection")
        client.create_collection(
            collection_name=new_collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dim, 
                distance=models.Distance.COSINE),
            init_from=models.InitFrom(collection=collection_name)
    )
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

def get_qdrant_vectorstore(client, embeddings, collection_name):
    embeddings = embeddings
    vector_store = Qdrant(
    client=client, collection_name=collection_name, 
    embeddings=embeddings,
)
    return vector_store