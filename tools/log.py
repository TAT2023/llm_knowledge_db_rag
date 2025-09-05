import logging

import os


from datetime import datetime

def get_loggger(log_dir):

    os.makedirs(log_dir,exist_ok=True)

    log_filename = f"{log_dir}/log_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger("debug_logger")

    logger.setLevel(logging.DEBUG)

    # 避免重复添加处理器，handlers列表存储了当前日志器（logger）已添加的处理器
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件处理器
    file_handler = logging.FileHandler(log_filename,encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 将处理器添加到日志器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = get_loggger(log_dir="./logs")
