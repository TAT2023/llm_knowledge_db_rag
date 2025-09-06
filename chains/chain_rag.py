


def Chain_RAG():
    """
    不带历史记录的知识库问答
    
    
    
    """


    # 基于召回结果和 query 结合构建的 promt 使用默认提升模板

    default_template_rq = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""

    def __init__(self,model:str,temperature:float=0.0,top_k:int=5,history_len:int,embedding:str,embedding_key:str=None):
        pass
