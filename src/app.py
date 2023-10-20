from dotenv import load_dotenv
import os
import pdb
from qdrant_client import QdrantClient
import streamlit as st
from utils.utils import *
from utils.htmlTemplates import css, bot_template, user_template
from streamlit import config
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema import ChatMessage

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
            # self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

def main():
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

        # Select a collection
        selected_collection = st.selectbox("Select a University", collection_names, index=None)

        if selected_collection:
            vectorstore = get_qdrant_vectorstore(client, selected_collection)
            st.success(f"Start chatting!")
        else:
            st.info("Please select a university name to contniue.")
            st.stop()

    msgs = StreamlitChatMessageHistory()

    qa_chain = get_conversation_chain(vectorstore, msgs)

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