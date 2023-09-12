import logging
import logging.config


from VideoChatGPT.settings import LOGGING_CONFIG

def setup_logging():
    """
    设置日志配置（Set up logging configuration）

    参数 (Parameters):
    无 (None)

    返回值 (Returns):
    无 (None)
    """
    # 应用日志配置（Apply the logging configuration）
    logging.config.dictConfig(LOGGING_CONFIG)

    # 初始化全局的日志记录器（Initialize the global logger）
    global global_logger
    global_logger = logging.getLogger("video_chat_gpt.global_logger")


def logger(message, log_level="info"):
    """
    自定义的日志记录器函数（Custom logger function）

    参数 (Parameters):
    - message (str): 要记录的消息内容（The message content to log）
    - log_level (str): 日志级别，默认为 'info'（The log level, default is 'info'）

    返回值 (Returns):
    无 (None)
    """

    # 获取与指定日志级别对应的日志函数（Get the logging function corresponding to the specified log level）
    log_func = getattr(global_logger, log_level.lower(), None)

    # 判断获取到的日志函数是否有效（Check if the obtained logging function is valid）
    if log_func is not None:
        # 调用日志函数，记录消息（Call the logging function to log the message）
        log_func(message)
    else:
        # 如果日志级别无效，记录错误消息（If the log level is invalid, log an error message）
        global_logger.error(f"无效的日志级别：{log_level} (Invalid log level: {log_level})")
