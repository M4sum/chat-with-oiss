from dotenv import load_dotenv
import os
import pdb
from qdrant_client import QdrantClient
import streamlit as st
from utils.utils import *
from utils.htmlTemplates import css, bot_template, user_template
from streamlit import config


def handle_userinput(user_question):
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
    except TypeError as e:
        st.warning("Please enter a valid URL in the sidebar to get started.")
        return
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your Northwestern OISS website",
                       page_icon=":robot_face:")
    client = QdrantClient(
            url=os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
    st.write(css, unsafe_allow_html=True)

    # with st.container():
    #     st.write("This website only works for Northwestern University's OISS website currently. \
    #          We are working on adding more universities")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your OISS website :robot_face:")
    
    user_question = st.text_input("Ask any immigration question to your school's OISS website:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        # Get all collections
        st.subheader("Universities supported")
        collection_names = get_qdrant_collections(client)

        # Select a collection
        selected_collection = st.selectbox("Select a University", collection_names, index=None)

        # # Create a new collection if selected option is "Create new collection"
        # if selected_collection == "Create new collection":
        #     new_collection_name = st.text_input("Enter a University name")
        #     url = st.text_input("Enter the URL of the University's OISS website")
        #     if st.button("Create"):
        #         if new_collection_name and url:
        #             with st.spinner("Processing"):
        #                 create_qdrant_collection(client, new_collection_name)
        #                 raw_text = get_documents(url)
        #                 text_chunks = get_text_chunks(raw_text)
        #                 vectorstore = get_qdrant_vectorstore(client, new_collection_name)
        #                 vectorstore.add_texts(text_chunks)
        #                 st.session_state.conversation = get_conversation_chain(vectorstore)
        #                 st.success(f"Collection {new_collection_name} created successfully")
        #         else:
        #             st.warning("Please enter a University name and OISS website url")
        # else:
        if selected_collection:
            vectorstore = get_qdrant_vectorstore(client, selected_collection)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success(f"Start chatting!")


if __name__ == '__main__':
    main()