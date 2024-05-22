import base64
import re
import os
import streamlit as st
from streamlit_chat import message
from pdfquery import PDFQuery

st.set_page_config(page_title="证券期货业数据治理问答")

def get_binary_file_downloader_html(bin_file, file_label='File'):
    href=""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}"> {file_label}</a>'
    finally:
        return href

def display_messages():
    # st.subheader("Chat")
    for i, (msgs, is_user) in enumerate(st.session_state["messages"]):
        if is_user :
            message(msgs, is_user=is_user, key=str(i))
        else:
            if (len(msgs[1]) > 0):
                with st.expander(f"已经找到{len(msgs[1])}篇参考资料点击这里查看详细信息"):
                    pattern = re.compile(r"'source': '(.+?)', 'page': (\d+)")
                    for item in msgs[1]:
                        # Using regex to find all matches in the string
                        print(item)
                        matches = pattern.findall(item)
                        for match in matches:
                            source, page = match
                            file_name = source.split('\\\\')[-1]

                            # Creating markdown link
                            source = os.getcwd()+'/RAG-FILES/' + file_name

                            file_path = source
                            file_label = f"{file_name} - Page {page}"
                            link_str = get_binary_file_downloader_html(file_path, file_label)
                            if link_str != "":
                                st.markdown(get_binary_file_downloader_html(file_path, file_label),
                                        unsafe_allow_html=True)
                message(msgs[0], is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            query_temp = st.session_state["pdfquery"].ask(user_text)
            query_text = (query_temp["answer"]).replace('\n', ' ')
            query_docs = query_temp["docs"]

            query_docs_str = ",".join(str(doc) for doc in query_docs)

            pattern = r"metadata=\{(.*?)\}"
            # 使用正则表达式找到所有匹配的metadata
            sources = re.findall(pattern, query_docs_str)

        st.session_state["messages"].append((user_text, True))


        msgs = [query_text, sources]
        st.session_state["messages"].append((msgs, False))



def main():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
        st.session_state["pdfquery"] = PDFQuery()

    st.header("证券期货业数据治理问答")

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input",  on_change=process_input)


if __name__ == "__main__":
    main()