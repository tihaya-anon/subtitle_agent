import logging
import sys
from config.settings import LOG_FILE_PATH

def setup_logger():
    """配置全局 logger"""
    
    # 获取根 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 防止重复添加 handler (如果模块被重载)
    if logger.hasHandlers():
        # 检查是否已经有了我们的自定义 handler
        # 这对于 GUI 重载尤其重要
        if any(isinstance(h, logging.FileHandler) and h.baseFilename == str(LOG_FILE_PATH) for h in logger.handlers):
             return logger
        # 如果没有，清空 handlers 重新配置 (例如在 PySide6 重载时)
        for h in logger.handlers[:]:
             logger.removeHandler(h)

    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - (%(module)s.%(funcName)s:%(lineno)d) - %(message)s'
    )

    # 1. 输出到控制台 (StreamHandler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 2. 输出到文件 (FileHandler)
    try:
        file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except IOError as e:
        # 如果文件句柄被占用，这可能会发生
        print(f"警告: 无法设置日志文件处理器: {e}")
        pass # 至少控制台 logger 还能工作

    logger.info("Logger 已初始化。")
    return logger

# 立即初始化并获取实例，以便其他模块可以直接导入 logger
logger = setup_logger()