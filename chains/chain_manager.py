
from langchain.schema import HumanMessage, AIMessage
from chains.chain_chat_llm_with_history import Chain_chat_llm_with_history
from chains.chain_rag import Chain_RAG
from chains.chain_rag_with_history import Chain_RAG_with_history
from tools.log import logger


def convert_memory_to_gradio(langchain_messages):
    """
    将 LangChain 的消息列表转换为 Gradio Chatbot 格式
    """
    gradio_messages = []
    
    # 遍历所有消息，将连续的用户消息和AI消息配对
    i = 0
    while i < len(langchain_messages):
        # 检查当前消息和下一个消息是否是一对(用户, AI)
        if (i + 1 < len(langchain_messages) and 
            isinstance(langchain_messages[i], HumanMessage) and 
            isinstance(langchain_messages[i+1], AIMessage)):
            
            # 添加一对消息
            user_msg = langchain_messages[i].content
            bot_msg = langchain_messages[i+1].content
            gradio_messages.append((user_msg, bot_msg))
            i += 2  # 跳过这一对消息
            
        else:
            # 处理不成对的消息（理论上不应该发生，但安全起见）
            if isinstance(langchain_messages[i], HumanMessage):
                gradio_messages.append((langchain_messages[i].content, None))
            elif isinstance(langchain_messages[i], AIMessage):
                gradio_messages.append((None, langchain_messages[i].content))
            i += 1
            
    return gradio_messages

class Chain_Manager:
    """
    管理不同的问答链
    """
    def __init__(self):
        self.chain_rag = {}
        self.chain_rag_with_history = {}
        self.chain_llm_with_history = {}

    def chain_rag_answer(self,question:str,model:str="openai",embedding:str="m3e",temperature:float=0.0,top_k:int=5):
        try:
            if (model,embedding) not in self.chain_rag:
                logger.info(f"创建新的无历史记录问答链: model={model}, embedding={embedding}")
                self.chain_rag[(model,embedding)] = Chain_RAG(model=model,embedding=embedding,temperature=temperature,top_k=top_k)
            chain = self.chain_rag[(model,embedding)]
            response = chain.answer(question=question,temperature=temperature,top_k=top_k)
            return "",response

        except Exception as e:
            logger.error(f"调用无历史记录问答链失败: {str(e)}",exc_info=True)
            return "",f"chain_rag error: {str(e)}"

    def chain_rag_with_history_answer(self,question:str,chat_history:list=[],model:str="openai",embedding:str="m3e",temperature:float=0.0,top_k:int=5):
        try:
            if (model,embedding) not in self.chain_rag_with_history:
                logger.info(f"创建新的有历史记录问答链: model={model}, embedding={embedding}")
                self.chain_rag_with_history[(model,embedding)] = Chain_RAG_with_history(model=model,embedding=embedding,temperature=temperature,top_k=top_k,chat_history=chat_history)
            chain = self.chain_rag_with_history[(model,embedding)]
            chat_history = chain.answer(question=question,temperature=temperature,top_k=top_k)
            return "",convert_memory_to_gradio(chat_history)

        except Exception as e:
            logger.error(f"调用有历史记录问答链失败: {str(e)}",exc_info=True)
            return "",f"chain_rag_with_history error: {str(e)}"

    def chain_llm_answer(self,question:str,model:str="openai",temperature:float=0.0):
        try:
            if model not in self.chain_llm_with_history:
                logger.info(f"创建新的纯llm问答链: model={model}")
                self.chain_llm_with_history[model] = Chain_chat_llm_with_history(model=model,temperature=temperature)
            chain = self.chain_llm_with_history[model]
            chat_history = chain.answer(question=question,temperature=temperature)
            return "",convert_memory_to_gradio(chat_history)
        except Exception as e:
            logger.error(f"调用纯llm问答链失败: {str(e)}",exc_info=True)
            return "",f"chain_llm error: {str(e)}"
    
    def clear_all_history(self):
        """清除所有问答链的历史记录"""
        for key, chain in self.chain_rag_with_history.items():
            chain.clear_history()
            logger.info(f"清除有历史记录问答链 {key} 的历史记录")
        
        for key, chain in self.chain_llm_with_history.items():
            chain.clear_history()
            logger.info(f"清除纯llm问答链 {key} 的历史记录")
        
        return []
