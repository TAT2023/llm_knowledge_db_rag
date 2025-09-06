import gradio as gr

from dotenv import load_dotenv, find_dotenv

from chains.chain_rag import Chain_RAG
from chains.chain_rag_with_history import Chain_RAG_with_history
from tools.log import logger


# 从 .env 文件中读取环境变量避免将敏感信息（如API密钥）硬编码到代码中
_=load_dotenv(find_dotenv())



AIGC_AVATAR_PATH = "./figures/aigc_avatar.png"
DATAWHALE_AVATAR_PATH = "./figures/datawhale_avatar.png"
AIGC_LOGO_PATH = "./figures/aigc_logo.png"
DATAWHALE_LOGO_PATH = "./figures/datawhale_logo.png"



LLM_MODEL_DICT = {
    "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"],
    "wenxin": ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"],
    "xinhuo": ["Spark-1.5", "Spark-2.0"],
    "zhipuai": ["chatglm_pro", "chatglm_std", "chatglm_lite"]
}
LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()),[])
INIT_LLM = "chatglm_std"
EMBEDDING_MODEL_LIST = ['zhipuai', 'openai', 'm3e']
INIT_EMBEDDING_MODEL = "m3e"




class Chain_Manager:
    """
    管理不同的问答链
    """
    def __init__(self):
        self.chain_rag = {}
        self.chain_rag_with_history = {}
        self.chain_llm = {}

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
            return "",chat_history

        except Exception as e:
            logger.error(f"调用有历史记录问答链失败: {str(e)}",exc_info=True)
            return "",f"chain_rag_with_history error: {str(e)}"

def create_db_from_files(files, embeddings="m3e"):
    logger.info(f"start to create vector db from files: {files} with embeddings: {embeddings}")



chain_manager = Chain_Manager()

block = gr.Blocks()

with block as demo:
    with gr.Row(equal_height=True):
        gr.Image(value=AIGC_LOGO_PATH,scale=1,min_width=10,show_label=False,show_download_button=False,container=False)

        with gr.Column(scale=2):
            gr.Markdown("""<h1><center>私有知识库RAG</center></h1>
                <center>LLM-UNIVERSE</center>
                """)
        gr.Image(value=DATAWHALE_LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False, container=False)


    with gr.Row():
        with gr.Column(scale=4):
            
            #历史记录
            chatbot = gr.Chatbot(height=400, show_copy_button=True, show_share_button=True, avatar_images=(AIGC_AVATAR_PATH, DATAWHALE_AVATAR_PATH))

            #用户当前输入框
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_with_his_btn = gr.Button("Chat db with history")
                db_wo_his_btn = gr.Button("Chat db without history")
                llm_btn = gr.Button("Chat with llm")
            
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(components=[chatbot], value="Clear console")
        

        with gr.Column(scale=1):
            file = gr.File(label='请选择知识库目录', file_count='directory',
                           file_types=['.txt', '.md', '.docx', '.pdf'])
            with gr.Row():
                init_db = gr.Button("知识库文件向量化")
                # 设置初始化知识库按钮的点击事件，点击后调用 init_vector_store 函数，传入 file 组件的值，并更新 status 组件的内容。
            model_argument = gr.Accordion("参数配置", open=False)
            with model_argument:
                temperature = gr.Slider(0,
                                        1,
                                        value=0.01,
                                        step=0.01,
                                        label="llm temperature",
                                        interactive=True)

                top_k = gr.Slider(1,
                                  10,
                                  value=3,
                                  step=1,
                                  label="vector db search top k",
                                  interactive=True)

                history_len = gr.Slider(0,
                                        5,
                                        value=3,
                                        step=1,
                                        label="history length",
                                        interactive=True)

            model_select = gr.Accordion("模型选择")
            with model_select:
                llm = gr.Dropdown(
                    LLM_MODEL_LIST,
                    label="large language model",
                    value=INIT_LLM,
                    interactive=True)

                embeddings = gr.Dropdown(EMBEDDING_MODEL_LIST,
                                         label="Embedding model",
                                         value=INIT_EMBEDDING_MODEL)

        # 为知识库文件向量化按钮设置点击事件
        init_db.click(create_db_from_files,
                      inputs=[file, embeddings], outputs=[msg])
        
        db_with_his_btn(chain_manager.chain_rag_with_history_answer,
                        inputs=[msg, chatbot, llm, embeddings, temperature, top_k], outputs=[msg, chatbot])
        
        db_wo_his_btn(chain_manager.chain_rag_answer,
                      inputs=[msg, llm, embeddings, temperature, top_k], outputs=[msg, chatbot])

# 启动前关闭可能存在的其他 Gradio 应用实例
gr.close_all()

# gradio,启动！
demo.launch()


