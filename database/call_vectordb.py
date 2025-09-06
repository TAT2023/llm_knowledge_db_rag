
import os

from langchain.vectorstores import Chroma
from embedding.call_embedding import get_embedding
from tools.log import logger

def load_vectordb(persist_path:str,embedding):
    """
    加载向量数据库
    """
    return Chroma(
        persist_directory=persist_path,
        embedding_function=embedding
    )

def get_vectordb(persist_path:str=None,embedding=None):
    if isinstance(embedding,str):
        embedding = get_embedding(embedding)

    if os.path.exists(persist_path):
        contents = os.listdir(persist_path)
        if len(contents) > 0:
            return load_vectordb(persist_path,embedding)
    
    logger.error(f"向量数据库路径 {persist_path} 不存在或为空，请先创建向量数据库")
    return None


