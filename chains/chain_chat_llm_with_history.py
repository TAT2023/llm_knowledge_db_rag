

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
        self.llm = model_to_llm(model,self.temperature)

        self.chain = ConversationChain(
            llm=self.llm,
            memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
        )
        
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

        response = self.chain.run(input=question)
        logger.info(f"模型 {self.model} 回答: {response}")

        return self.chain.memory.chat_memory.messages