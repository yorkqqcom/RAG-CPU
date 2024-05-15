import os
import streamlit as st
from streamlit_chat import message
from pdfquery import PDFQuery

st.set_page_config(page_title="证券期货业数据治理问答")


def display_messages():
    # st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            query_temp = st.session_state["pdfquery"].ask(user_text)
            query_text = (query_temp["answer"]).replace('\n', ' ')
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((query_text, False))



def main():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
        st.session_state["pdfquery"] = PDFQuery()

    st.header("证券期货业数据治理问答")

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input",  on_change=process_input)

    # st.divider()
    # st.markdown("Source code: [Github](https://github.com/Anil-matcha/ChatPDF)")


if __name__ == "__main__":
    main()