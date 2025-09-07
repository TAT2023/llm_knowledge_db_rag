import os
import re

from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredFileLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools.log import logger

from embedding.call_embedding import get_embedding


DEFAULT_KNOWLEDGE_PATH = "./knowledge"
DEFAULT_PERSIST_PATH = "./vector_db/chroma"


def get_files(dir_path):
    file_list = []
    for filepath ,dirnames,filenames in os.walk(dir_path):
        for filename in filenames:
            file_list.append(os.path.join(filepath,filename))

def file_loader(file,loaders):
    if not os.path.isfile(file):
        [file_loader(os.path.join(file,f),loaders) for f in os.listdir(file)]
        return
    
    file_type = file.split(".")[-1]

    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file))
    elif file_type == 'md':
        pattern = r"不存在|风控"
        match = re.search(pattern,file)
        if not match:
            loaders.append(UnstructuredMarkdownLoader(file))
    
    elif file_type == 'txt':
        loaders.append(UnstructuredFileLoader(file))

def create_db(files=DEFAULT_KNOWLEDGE_PATH,persist_dir=DEFAULT_PERSIST_PATH,embedding="openai"):
    "根据知识文件创建向量数据库"

    if files is None:
        logger.error("路径为空")
        return "can't load empty path"
    #检查file类型
    if not isinstance(files,list):
        files = [files]
    
    # 检查文件是否存在
    for file in files:
        if not os.path.exists(file):
            logger.error(f"文件不存在: {file}")
            return f"file not exists: {file}"
        
    # 加载文件
    loaders = []

    try:
        for file in files:
            file_loader(file,loaders)
        

        # 加载文件内容
        docs = []
        for loader in loaders:
            if loader is not None:
                # loader.load() 返回的是一个列表
                docs.extend(loader.load())

        if len(docs) == 0:
            logger.error("没有加载到任何文档")
            return "no document loaded"
        
        logger.info(f"加载了 {len(docs)} 个文档")

    except Exception as e:
        # exc_info=True 会打印完整的堆栈信息
        logger.error(f"加载文件时出错：{str(e)}",exc_info=True)
        return f"file load error: {str(e)}"
    

    # 切分文档
    try:
        # chunk_size: 每个小块的最大长度
        # chunk_overlap: 小块之间的重叠长度
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150
        )
        split_docs = text_splitter.split_documents(docs)

        if len(split_docs) == 0:
            logger.error("没有切分出任何文档")
            return "no split document"

        logger.info(f"切分后的文档数量：{len(split_docs)}")
        
    except Exception as e:
        logger.error(f"切分文档时出错：{str(e)}",exc_info=True)
        return f"document split error: {str(e)}"
    
    # 加载 Embedding 模型实例
    try:
        if isinstance(embedding,str):
            embedding = get_embedding(embedding)
        logger.info(f"使用的 Embedding 模型: {embedding}")
    except Exception as e:
        logger.error(f"初始化 Embedding 模型时出错：{str(e)}",exc_info=True)
        return f"load embedding model error: {str(e)}"
    
    # 确保持久化路径存在
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    
    # 创建向量数据库ChromaDB

    try:
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding,
            persist_directory=persist_dir
        )
        # 验证数据库是否创建成功
        vector_count = vectordb._collection.count()

        if vector_count == 0:
            logger.error("向量数据库创建失败，未存储任何向量")
            return "vector db create error"
        logger.info(f"向量数据库创建成功，存储了 {vector_count} 个向量")

        # 持久化数据库
        vectordb.persist()
        logger.info(f"向量数据库已持久化到: {persist_dir}")
        return vectordb
    except Exception as e:
        logger.error(f"创建向量数据库时出错：{str(e)}",exc_info=True)
        return f"vector db create error: {str(e)}"


if __name__ == "__main__":
    create_db(embedding="m3e")