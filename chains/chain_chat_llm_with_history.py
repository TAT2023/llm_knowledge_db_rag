
import time
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from tools.log import logger

from llm.model_to_llm import model_to_llm

class Chain_chat_llm_with_history:
    """
    带历史记录的纯LLM问答
    """

    def __init__(self,model:str,temperature:float=0.0):
        self.model = model
        self.temperature = temperature
        
        try:
            self.llm = model_to_llm(model,self.temperature)
            logger.info(f"创建Chain_chat_llm_with_history时加载LLM模型 {model} 成功")
        except Exception as e:
            logger.error(f"创建Chain_chat_llm_with_history时加载LLM模型 {model} 失败: {str(e)}",exc_info=True)
            raise
        
        try:
            self.chain = ConversationChain(
                llm=self.llm,
                memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
            )
            logger.info(f"创建Chain_chat_llm_with_history问答链成功")
        except Exception as e:
            logger.error(f"创建Chain_chat_llm_with_history问答链失败: {str(e)}",exc_info=True)
            raise
        
    def clear_history(self):
        """清除历史记录"""
        self.chain.memory.clear()

    def answer(self,question:str=None,temperature:float=None):
        """
        调用问答链回答
        """

        if question is None or question.strip() == "":
            logger.error("问题不能为空，请重新输入！")
            return "问题不能为空，请重新输入！", []

        if temperature is not None:
            temperature = self.temperature

        start_time = time.time()
        response = self.chain.run(input=question)
        logger.info(f"Chain_chat_llm_with_history调用完成,耗时 {time.time()-start_time:.2f} 秒 | answer: {response}")

        return self.chain.memory.chat_memory.messages