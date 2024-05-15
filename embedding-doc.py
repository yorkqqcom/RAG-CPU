from langchain.text_splitter import CharacterTextSplitter

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
run_path = os.getcwd()
EMBEDDING_MODEL = 'bge'
EMBEDDING_DEVICE = "cpu"

text_splitter = CharacterTextSplitter(
    separator="。",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
# create embeddings
# embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL],
#                                    model_kwargs={'device': EMBEDDING_DEVICE})
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",
                                   cache_folder=run_path,
                                   model_kwargs={'device': EMBEDDING_DEVICE})

# 指定要列出文件的目录
directory_path = run_path + '\\RAG-FILES\\'

# 获取目录中的文件和文件夹列表
entries = os.listdir(directory_path)
files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
file_num = 1

for file in files:
    try:
        # print(file)
        loader = PyPDFLoader(directory_path + file)  # Title and NarrativeText
        docs_all = loader.load()  # 这里加载文档。
        pages = loader.load_and_split(text_splitter)

        knowledge_base = FAISS.from_documents(pages, embeddings)

        file_dir = "./file-index/file_"
        file_index = file_dir + str(file_num) + ".idx"
        FAISS.save_local(knowledge_base, file_index)
        file_num += 1
        print(file_index, '-', file)
    except Exception as e:
        print(f"发生了一个错误：{e}")
        continue
        # 如果发生其他异常，则执行以下代码
        # print(f"发生了一个错误：{e}")