import logging
import sys
import os

# 全局统一的日志格式字符串：所有 Handler 共用，避免在 setup_logging / add_file_handler / main.py 三处各写一遍。
LOG_FORMAT = '%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s'


def build_formatter():
    return logging.Formatter(LOG_FORMAT)


def setup_logging(log_path=None, level=logging.INFO):
    """
    配置全局日志记录器。
    - 输出到控制台 (stdout)。
    - 如果提供了路径，则输出到文件（目录不存在时会自动创建）。
    """
    log_formatter = build_formatter()
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除任何可能已经存在的处理器
    root_logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    if log_path:
        # 干净地补齐目录，避免因父目录缺失而静默跳过文件日志
        log_dir = os.path.dirname(log_path)
        if log_dir:
            try:
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
            except OSError:
                log_dir = None
        if log_dir:
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(log_formatter)
            root_logger.addHandler(file_handler)

    logging.info("Logger initialized.")


def add_file_handler(logger, log_file_path):
    """
    为已存在的logger动态添加文件处理器。
    每次添加前，先清理掉旧的 FileHandler，防止新任务的日志残留在上一次任务的输出文件夹中。
    """
    # 清理所有既有的 FileHandler
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    log_formatter = build_formatter()

    # 确保目录存在
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Log file handler added for: {log_file_path}")