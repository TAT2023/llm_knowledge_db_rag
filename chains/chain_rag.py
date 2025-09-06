
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from database.call_vectordb import get_vectordb
from llm import model_to_llm

from tools.log import logger
import time

class Chain_RAG:
    """
    不带历史记录的知识库问答
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - embedding：使用的embedding模型名称
    - template：使用的prompt模板
    """


    # 基于召回结果和 query 结合构建的 promt 使用默认提升模板

    default_template_rq = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""



    def __init__(self,model:str,temperature:float=0.0,top_k:int=5,embedding:str="openai",template=default_template_rq):
        
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.embedding = embedding
        self.template = template
        self.llm = model_to_llm(model,self.temperature)
        self.vector_db = get_vectordb(embedding=embedding)


        self.chain_prompt = PromptTemplate(
            template=self.template,
            input_variables=["context","question"]
        )

        self.retriever = self.vector_db.as_retriever(search_type="similarity",search_kwargs={"k":self.top_k})


        self.chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt":self.chain_prompt}
        )
    
    def answer(self,question:str=None,temperature:float=None,top_k:int=5):
        """
        - question: 用户输入的问题
        - temperature: 温度系数，控制生成的随机性
        - top_k: 返回检索的前k个相似文档
        """
        if question is None or question.strip() == "":
            return "问题不能为空，请重新输入！",[]

        if temperature is None:
            temperature = self.temperature
        
        if top_k is None:
            top_k = self.top_k

        start_time = time.time()
        try:
            logger.info(f"Chain_RAG开始调用answer | question: {question} | temperature: {temperature} | top_k: {top_k}")
            result = self.chain({"query":question})
            answer = result['result']
            source_docs = result['source_documents']
            logger.info(f"Chain_RAG调用answer成功 | answer: {answer} | source_docs count: {len(source_docs)} | time cost: {time.time() - start_time}")
        
        except Exception as e:
            logger.error(f"Chain_RAG调用answer失败 | error: {str(e)}",exc_info=True)
            return f"Chain_RAG调用answer失败: {str(e)}",[]
        return answer