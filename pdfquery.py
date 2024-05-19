import json
import os
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from typing import List
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from Qwen import Qwen

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
run_path = os.getcwd()
EMBEDDING_MODEL = 'bge'
EMBEDDING_DEVICE = "cpu"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",
                                   cache_folder=run_path,
                                   model_kwargs={'device': EMBEDDING_DEVICE})

def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string.:"""
    # formatted = [
    #     f"Article Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
    #     for doc in docs
    # ]
    formatted = [
        f"Article Title:{doc.metadata['source']} \nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)

class PDFQuery:

    def __init__(self) -> None:
        # 设置目录名称
        directory = "./file-index/"
        # 列出目录中所有文件
        dir_list = os.listdir(directory)
        dirs = [f for f in dir_list if f.endswith(".idx")]

        # file_name = './file-index/file_2.idx'
        loaders = [
            FAISS.load_local(os.path.join(directory, dir), embeddings, allow_dangerous_deserialization=True) for
            dir in dirs]

        load_merge = None
        for loader in loaders:
            if load_merge is None:
                load_merge = loader
            else:
                load_merge.merge_from(loader)
        retriever = load_merge.as_retriever()
        # llm = Qwen()
        llm = ChatOpenAI()
        prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following retrieved context to answer the question in an expert manner. If you don’t know the answer, just say that you don’t know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
)

        format = itemgetter("docs") | RunnableLambda(format_docs)
        answer = prompt | llm | StrOutputParser()

        self.rag_chain = (
            RunnableParallel(question=RunnablePassthrough(), docs=retriever)
            .assign(context=format)
            .assign(answer=answer)
            .pick(["answer", "docs"])
        )
    def ask(self, question: str) -> str:

        if self.rag_chain is None:
            response = "Please, add a document."
        else:
            # docs = self.db.get_relevant_documents
            response = self.rag_chain.invoke(question)
            # response = self.chain.run(input_documents=docs, question=question)
        return response

    def ask2(self, question: str) -> str:
        msg = ""
        return msg