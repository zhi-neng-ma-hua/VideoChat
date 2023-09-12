from pathlib import Path

# 获取当前文件的父目录的父目录作为基础目录（Get the parent's parent directory of the current file as the base directory）
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据集的路径（Path for the dataset）
DATA_PATH = Path(BASE_DIR) / "dataset"

# 日志文件的存储路径（Path for storing log files）
LOG_PATH = Path(BASE_DIR) / "logs"

# HDF5 图像数据文件的名称（Name of the HDF5 image data file）
HDF5_FILE_NAME = "image.h5"

# 词汇表文件的名称（Name of the vocabulary file）
VOCABULARY_FILE_NAME = "vocabulary.json"

# 图像数据文件的完整路径（Full path for the image data file）
IMAGE_DATA_PATH = DATA_PATH / HDF5_FILE_NAME

# 词汇表数据文件的完整路径（Full path for the vocabulary data file）
VOCABULARY_DATA_PATH = DATA_PATH / VOCABULARY_FILE_NAME

# 用于计算的 CUDA 设备（CUDA devices for computing）
CUDA_DEVICES = "0,1,2,3"

# 日志配置字典（Logging configuration dictionary）
LOGGING_CONFIG = {
    "version": 1,  # 日志配置的版本，通常为 1（The version of logging config, usually 1）
    "disable_existing_loggers": False,  # 是否禁用所有现有的日志记录器（Whether to disable all existing loggers）

    "formatters": {  # 格式化器设置（Formatter settings）
        "standard": {  # 格式化器的名称（Name of the formatter）
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"  # 日志输出格式（Log output format）
        },
    },

    "handlers": {  # 处理器设置（Handler settings）
        "default": {  # 默认处理器（Default handler）
            "level": "INFO",  # 日志级别（Logging level）
            "formatter": "standard",  # 使用的格式化器（Formatter to use）
            "class": "logging.StreamHandler",  # 处理器类，这里是输出到控制台（Handler class, here it is output to console）
        },
        "info_file": {  # 信息日志文件处理器（Info log file handler）
            "level": "INFO",  # 日志级别（Logging level）
            "formatter": "standard",  # 使用的格式化器（Formatter to use）
            "class": "logging.FileHandler",  # 处理器类，这里是输出到文件（Handler class, here it is output to file）
            "filename": LOG_PATH / "info.log",  # 输出的文件路径（Output file path）
        },
        "error_file": {  # 错误日志文件处理器（Error log file handler）
            "level": "ERROR",  # 日志级别（Logging level）
            "formatter": "standard",  # 使用的格式化器（Formatter to use）
            "class": "logging.FileHandler",  # 处理器类，这里是输出到文件（Handler class, here it is output to file）
            "filename": LOG_PATH / "error.log",  # 输出的文件路径（Output file path）
        },
        "debug_file": {  # 调试日志文件处理器（Debug log file handler）
            "level": "DEBUG",  # 日志级别（Logging level）
            "formatter": "standard",  # 使用的格式化器（Formatter to use）
            "class": "logging.FileHandler",  # 处理器类，这里是输出到文件（Handler class, here it is output to file）
            "filename": LOG_PATH / "debug.log",  # 输出的文件路径（Output file path）
        }
    },

    "loggers": {  # 记录器设置（Logger settings）
        "": {  # 默认记录器（Default logger）
            "handlers": ["default", "info_file", "error_file", "debug_file"],  # 使用的处理器列表（List of handlers to use）
            "level": "INFO",  # 日志级别（Logging level）
            "propagate": True  # 是否传播日志流（Whether to propagate the log stream）
        }
    }
}
