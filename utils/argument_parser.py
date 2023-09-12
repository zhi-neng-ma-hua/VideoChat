import argparse

from VideoChatGPT.settings import DATA_PATH, IMAGE_DATA_PATH, VOCABULARY_DATA_PATH


class ArgumentParser(object):

    def __init__(self):
        # 初始化参数解析器（Initialize the argument parser）
        self.parser = argparse.ArgumentParser(description="training.py")

        self.add_arguments()

        # 解析参数（Parse the arguments）
        self.args = self.parser.parse_args()

    def add_arguments(self):
        args = [
            # Todo ========== 模型参数（Model Parameters） ==========
            # 设置嵌入层的大小（Set the size of the embedding layer）
            ("-n_emb", int, 512, "Embedding size"),
            # 设置隐藏层的大小（Set the size of the hidden layer）
            ("-n_hidden", int, 512, "Hidden size"),
            # 设置注意力头的数量（Set the number of attention heads）
            ("-n_head", int, 8, "Number of head"),
            # 设置模型中块的数量（Set the number of blocks in the model）
            # Todo ========== 文本和时间参数（Text and Time Parameters） ==========
            ("-n_block", int, 6, "Number of block"),
            # 设置文本输入的最大长度（Set the maximum length for text input）
            ("-max_len", int, 20, "Limited length for text"),
            # 设置模型的时间范围（Set the time range for the model）
            ("-time_range", int, 5, "Time range"),
            # 设置每秒最大评论数（Set the maximum number of comments per second）
            ("-max_cnum", int, 15, "Max comments each second"),
            # 设置波束搜索的波束大小（1 表示贪婪搜索）（Set the beam size for beam search (1 means greedy search)）
            ("-beam_size", int, 1, "Bean size"),
            # Todo ========== 训练参数（Training Parameters） ==========
            # 设置训练的批量大小（Set the batch size for training）
            ("-batch_size", int, 32, "Batch size"),
            # 设置训练周期的数量（Set the number of training epochs）
            ("-epoch", int, 100, "Number of epoch"),
            # 设置丢弃率（Set the dropout rate）
            ("-dropout", float, 0.2, "Dropout rate"),
            # 设置学习率（Set the learning rate）
            ("-lr", float, 1e-3, "Learning rate"),
            # 设置正则化的权重衰减（Set the weight decay for regularization）
            ("-weight_decay", float, 0.001, "Weight decay for regularization"),
            # 设置早停准则（Set the early stopping criteria）
            ("-early_stop", float, 20, "Early Stop"),
            # Todo ========== 数据和输出路径（Data and Output Paths） ==========
            # 设置数据（字典和图像）的路径（Set the path for data (dictionary and images)）
            ("-data_path", str, DATA_PATH, "dict and image path"),
            ("-image_data_path", str, IMAGE_DATA_PATH, "image path"),
            ("-vocabulary_data_path", str, VOCABULARY_DATA_PATH, "vocabulary path"),
            # 设置输出目录路径（Set the output directory path）
            ("-out_path", str, None, "out path"),
            # 设置生成结果的输出文件名（Set the output file name for generated results）
            ("-outfile", str, "out.json", "outfile for generation"),
            # 设置恢复预训练模型的路径（Set the path for restoring a pre-trained model）
            ("-restore", str, None, "Restoring model path"),
            # Todo ========== 模式（例如，训练、测试等）（Mode (e.g., train, test, etc.)） ==========
            ("-mode", str, None, "Mode (e.g., train, test, etc.)")
        ]

        for arg, default_type, default_value, help_text in args:
            self.parser.add_argument(arg, type=default_type, default=default_value, help=help_text)
