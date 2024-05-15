可在CPU运行的本地RAG，采用Qwen0.5B和bge-small-zh-v1.5，
配置要求：个人电脑，8G内存以上。

部署步骤

安装所需要的包 

pip install -r requirement.txt

下载模型

$env:HF_ENDPOINT = "https://hf-mirror.com"

huggingface-cli download --resume-download Qwen/Qwen1.5-0.5B-Chat --local-dir Qwen1.5-0.5B-Chat

运行程序

$env:HF_ENDPOINT = "https://hf-mirror.com"

streamlit run .\appUI.py