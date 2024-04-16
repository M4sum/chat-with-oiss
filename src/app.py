from cryptography.fernet import Fernet
from crud_collections import get_qdrant_collections, get_qdrant_vectorstore
from dotenv import load_dotenv
import os
import pdb
from qdrant_client import QdrantClient
import streamlit as st
from utils.utils import *
from utils.htmlTemplates import css, bot_template, user_template
from streamlit import config
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


chat_model_envvars = {
    "GPT-3.5": "OPENAI_API_KEY",
    "Gemini": "GOOGLE_API_KEY",
    "Command-R": "COHERE_API_KEY",
    "Claude": "ANTHROPIC_API_KEY"
}

user_key_prompt = "Enter your API key to get started. Keep it safe, as it'll be your key to coming back. \
    \n\n**Friendly reminder:** Chat with OISS works best with pay-as-you-go API keys. \
    Free trial API keys are limited to few requests a minute, not always enough to chat with assistants. \
        For more information on API rate limits, check respective API or pricing pages."
user_key_failed = "You entered an invalid API key."
user_key_success = "Thanks for signing in! Make sure to keep the API key safe, as it'll be your key to using this again.!"
api_key_placeholder = "Paste your OpenAI API key here (sk-...)"

key = Fernet.generate_key()



class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # pdb.set_trace()
        for idx, doc in enumerate(documents):
            # source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**source excerpt** {doc.page_content}")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

def main():
    show_chat=False
    load_dotenv()
    st.set_page_config(page_title="Chat with your Northwestern OISS website",
                       page_icon=":robot_face:")
    client = QdrantClient(
            url=os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
    st.write(css, unsafe_allow_html=True)
    with st.sidebar:
        # Get all collections
        st.subheader("Only supports Northwestern University currently")
        collection_names = get_qdrant_collections(client)
        embeddings = FastEmbedEmbeddings()
        # Select a collection
        # st.write(collection_names)
        selected_collection = st.selectbox("Select a University", [collection_name for collection_name in collection_names if collection_name == "Northwestern University"], index=None)
        selected_model = st.selectbox("Choose a chat model:", [model["name"] for model in chat_models], index=None)            

        if not (selected_collection and selected_model):
            st.info("Please select a university name and a chat model to contniue.")
            st.stop()
        else:
            st.info(user_key_prompt)
            selected_model_info = next(model for model in chat_models if model["name"] == selected_model)
            user_api_key = st.text_input(label=f"Enter your {selected_model_info['provider']} API key ({selected_model_info['api_key_link']}):", \
                             autocomplete="current-password", \
                                placeholder=api_key_placeholder,
                            )
            vectorstore = get_qdrant_vectorstore(client, embeddings, selected_collection)
            if user_api_key:
                api_key = encrypt_api_key(user_api_key, key)
                # st.write(api_key)
                if api_key:
                    store_api_key_in_session(api_key, chat_model_envvars[selected_model], key)
                    st.success("API Key successfully stored!")
                    show_chat=True
                    st.success(f"Start chatting!")
                else:
                    st.error("Please enter an API Key.")
                    st.stop()

    if show_chat:
        msgs = StreamlitChatMessageHistory()
        qa_chain = get_conversation_chain(vectorstore, selected_model, msgs)
        greet_message = "Hello, I am an AI Assistant tasked with answering your immigration related queries, from the Northwestern website. \
            I can help you with 3 kinds of tasks.\n \
                1. Help with you case\n \
                    2. Find a topic from OISS website\n \
                        3. Ask a general query\n \
                            Which of this would you require help with today?"
        
        if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
            msgs.clear()
            msgs.add_ai_message(greet_message)

        avatars = {"human": "user", "ai": "assistant"}
        for msg in msgs.messages:
            st.chat_message(avatars[msg.type]).write(msg.content)

        if user_query := st.chat_input(placeholder="Ask me anything!"):
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                retrieval_handler = PrintRetrievalHandler(st.container())
                stream_handler = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])

if __name__ == '__main__':
    main()