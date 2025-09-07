
import os

from langchain.vectorstores import Chroma
from embedding.call_embedding import get_embedding
from tools.log import logger
from database.create_db import DEFAULT_PERSIST_PATH

def load_vectordb(persist_path:str,embedding):
    """
    加载向量数据库
    """
    return Chroma(
        persist_directory=persist_path,
        embedding_function=embedding
    )

def get_vectordb(persist_path:str=DEFAULT_PERSIST_PATH,embedding=None):
    if isinstance(embedding,str):
        embedding = get_embedding(embedding)

    if os.path.exists(persist_path):
        contents = os.listdir(persist_path)
        if len(contents) > 0:
            return load_vectordb(persist_path,embedding)
    
    logger.error(f"向量数据库路径 {persist_path} 不存在或为空，请先创建向量数据库")
    raise ValueError(f"向量数据库加载失败: {persist_path} not exists or empty, please create vector db first")


