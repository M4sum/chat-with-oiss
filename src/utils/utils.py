from cachetools import TTLCache
from cryptography.fernet import Fernet
from langchain_community.chat_models import ChatOpenAI, ChatCohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.runnables import RunnableLambda
import os
import pdb
import qdrant_client
from requests import Session
import streamlit as st

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


chat_models = [
    {
        "name": "GPT-3.5",
        "provider": "OpenAI",
        "api_key_link": "https://beta.openai.com/signup/",
        "free_trial": True,
    },
    {
        "name": "Claude",
        "provider": "Anthropic",
        "api_key_link": "https://www.anthropic.com/pricing",
        "free_trial": False,
    },
    {
        "name": "Gemini",
        "provider": "Google",
        "api_key_link": "https://developers.generativeai.google/setup",
        "free_trial": True
    },
    {
        "name": "Command-R",
        "provider": "Cohere AI",
        "api_key_link": "https://dashboard.cohere.ai/signup",
        "free_trial": True,
    },
    # Add more chat models as needed
]

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

def get_conversation_chain(vectorstore, selected_model, msgs=None):
    
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', chat_memory=msgs, return_messages=True)
    
    if selected_model == "GPT-3.5":
        llm = ChatOpenAI(
                model_name="gpt-3.5-turbo", temperature=0, streaming=True
        ),
    elif selected_model == "Command-R":
        llm = ChatCohere(
                model_name="command", temperature=0.75, streaming=True
        ),
    else:
        st.error("model API not yet compatible")

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm[0],
        retriever=vectorstore.as_retriever(),
        # condense_question_prompt=prompt,
        memory=memory,
        # return_source_documents=True
    )
    # pdb.set_trace()
    return conversation_chain

def get_vectorstore(text_chunks):
    embeddings = FastEmbedEmbeddings()
    
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to encrypt the API key
def encrypt_api_key(api_key, key):
    cipher_suite = Fernet(key)
    encrypted_key = cipher_suite.encrypt(api_key.encode())
    return encrypted_key

# Function to decrypt the API key
def decrypt_api_key(encrypted_key, key):
    cipher_suite = Fernet(key)
    decrypted_key = cipher_suite.decrypt(encrypted_key).decode()
    return decrypted_key

# Function to store API key in session state
def store_api_key_in_session(api_key, env_var_name, key):
    os.environ[env_var_name] = decrypt_api_key(api_key, key)

# Function to retrieve API key from session state
def get_api_key_from_session(env_var_name, key):
    encrypted_key = os.environ.get(env_var_name)
    if encrypted_key:
        return decrypt_api_key(encrypted_key, key)
    else:
        return None