

from langchain.chains import ConversationalRetrievalChain
import time
import re
from database.call_vectordb import get_vectordb
from llm.model_to_llm import model_to_llm
from tools.log import logger

class Chain_RAG_with_history:
    """
    带历史记录的知识库问答
    """



    def __init__(self,model:str,temperature:float=0.0,top_k:int=5,chat_history:list=[],embedding:str="openai"):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.embedding = embedding
        self.chat_history = chat_history
        self.llm = model_to_llm(model,self.temperature)
        self.vector_db = get_vectordb(embedding=embedding)
    
        self.retriever = self.vector_db.as_retriever(search_type="similarity",search_kwargs={"k":top_k})

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )


    def clear_history(self):
        """清除历史记录"""
        self.chat_history.clear()


    def change_history_length(self,new_length:int=1):
        """调整历史记录长度"""

        if new_length < 0:
            logger.error("历史记录长度不能为负数")
            return
        elif new_length < len(self.chat_history):
            self.chat_history = self.chat_history[-new_length:]
            logger.info(f"历史记录长度调整为 {new_length}，当前历史记录数量为 {len(self.chat_history)}")
        else:
            logger.info(f"历史记录长度无需裁剪")

    def answer(self,question:str=None,temperature:float=None,top_k:int=5):
        """
        调用问答链回答
        """

        if question is None or question.strip() == "":
            logger.error("问题不能为空，请重新输入！")
            return "问题不能为空，请重新输入！", []
        
        if temperature is not None:
            temperature = self.temperature

        if top_k is None:
            top_k = self.top_k

        start_time = time.time()

        try:
            logger.info(f"Chain_RAG_with_history开始调用answer | question: {question} | temperature: {temperature} | top_k: {top_k} | chat_history_len: {len(self.chat_history)}")
            result = self.chain({"question":question,"chat_history":self.chat_history})
            answer = result["answer"]

            # 将换行符替换为 HTML 的 <br/> 标签，便于在网页中显示
            answer = re.sub(r"\\n", '<br/>', answer)

            source_docs = result.get("source_documents",[])
            logger.info(f"Chain_RAG_with_history调用完成，耗时 {time.time()-start_time:.2f} 秒 | answer: {answer} | source_docs_len: {len(source_docs)}")
            # 更新历史记录
            self.chat_history.append((question, answer))
            return self.chat_history
        except Exception as e:
            logger.error(f"Chain_RAG_with_history调用answer失败 | error: {str(e)}",exc_info=True)
            return f"Chain_RAG_with_history调用answer失败: {str(e)}", []
        