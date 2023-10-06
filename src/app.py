from dotenv import load_dotenv
import pdb
import streamlit as st
from utils.utils import get_documents, get_text_chunks, get_vectorstore, get_conversation_chain
from utils.htmlTemplates import css, bot_template, user_template
from streamlit import config


def handle_userinput(user_question):
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
    except TypeError as e:
        st.warning("Please enter a valid URL in the sidebar to get started.")
        return
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your Northwestern OISS website",
                       page_icon=":scales:")
    
    st.write(css, unsafe_allow_html=True)

    with st.container():
        st.write("This website only works for Northwestern University's OISS website currently. \
             We are working on adding more universities")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your OISS website :scales:")
    user_question = st.text_input("Ask any immigration question to your school's OISS website:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        url = st.text_input(
            "Give me the URL of your university's OISS website")
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_documents(url)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()