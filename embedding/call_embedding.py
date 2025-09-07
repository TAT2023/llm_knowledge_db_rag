from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings

def get_embedding(embedding:str,embedding_key:str=None):
    if embedding == 'm3e':
        return HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")

    elif embedding == 'openai':
        return OpenAIEmbeddings(openai_api_key=embedding_key)
    # elif embedding == 'zhipuai':
    #     return ZhipuAIEmbeddings(zhipuai_api_key=embedding_key)
    else:
        raise ValueError(f"不支持的embedding类型: {embedding}")
    