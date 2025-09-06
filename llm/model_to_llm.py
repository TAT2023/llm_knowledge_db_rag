

from dotenv import load_dotenv, find_dotenv
import os

from langchain_community.chat_models import QianfanChatEndpoint 
from langchain_community.chat_models import ChatSparkLLM
from langchain.chat_models import ChatOpenAI
from langchain.utils import get_from_dict_or_env

def parse_llm_api_key(model: str,env_file:dict()=None):
    """
    通过model名称提取API密钥
    """

    if env_file is None:
        _=load_dotenv(find_dotenv())
        env_file = os.environ
    
    if model == "openai":
        return get_from_dict_or_env(env_file, "OPENAI_API_KEY")
    
    elif model == "wenxin":
        return get_from_dict_or_env(env_file, "WENXIN_API_KEY", "WENXIN_SECRET_KEY")

    elif model == "spark":
        return get_from_dict_or_env(env_file, "SPARK_API_KEY", "SPARK_APPID", "SPARK_API_SECRET")

    elif model == "zhipuai":
        return get_from_dict_or_env(env_file, "ZHIPUAI_API_KEY", "ZHIPUAI_API_SECRET")



def model_to_llm(model:str,temperature:float=0.0):
    """
    星火:model,temperature,appid,api_key,api_secret
    百度问心:model,temperature,api_key,api_secret
    智谱:model,temperature,api_key
    OpenAI:model,temperature,api_key
    """
    llm = None

    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
        api_key = parse_llm_api_key("openai")
        llm = ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=api_key)

    elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
        api_key, Wenxin_secret_key = parse_llm_api_key("wenxin")
        llm = QianfanChatEndpoint(model_name=model, temperature=temperature, api_key=api_key, secret_key=Wenxin_secret_key)
    elif model in ["Spark-1.5", "Spark-2.0"]:
        api_key, appid, api_secret = parse_llm_api_key("spark")
        llm = ChatSparkLLM(model_name=model, temperature=temperature, api_key=api_key, appid=appid, secret_key=api_secret)
    elif model in ["chatglm_pro", "chatglm_std", "chatglm_lite"]:
        api_key = parse_llm_api_key("zhipuai")
        llm = QianfanChatEndpoint(model_name=model, temperature=temperature, api_key=api_key)
    else:
        raise ValueError(f"model{model} not support!!!")
    return llm
