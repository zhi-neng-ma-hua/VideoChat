import json
import numpy as np
import sys
import time
import torch

from contextlib import ExitStack
from itertools import islice
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from typing import Dict

from utils.decorators import timeit
from utils.logger_config import logger


class DatasetConfig:
    def __init__(self, data_path: str, vocabs: Dict[str, int], reverse_vocabs: Dict[int, str], images: Dict[str, int],
                 left_time_range: int, right_time_range: int, max_len: int, max_cnum: int, is_train: bool,
                 set_name: str = 'train'):
        self.data_path = data_path  # 数据路径（Data path）
        self.vocabs = vocabs  # 词汇表（Vocabulary）
        self.reverse_vocabs = reverse_vocabs  # 反向词汇表（Reverse vocabulary）
        self.images = images  # 图像数据（Image data）
        self.left_time_range = left_time_range  # 左侧时间范围（Left time range）
        self.right_time_range = right_time_range  # 右侧时间范围（Right time range）
        self.max_len = max_len  # 最大长度（Maximum length）
        self.max_cnum = max_cnum  # 最大评论数（Maximum number of comments）
        self.is_train = is_train  # 是否为训练集（Is it a training set）
        self.set_name = set_name  # 数据集名称（Dataset name）


class Dataset(TorchDataset):
    def __init__(self, config: DatasetConfig):
        self.data_path = config.data_path  # 数据路径（Data path）
        self.vocabs = config.vocabs  # 词汇表（Vocabulary）
        self.reverse_vocabs = config.reverse_vocabs  # 反向词汇表（Reverse vocabulary）
        self.images = config.images  # 图像数据（Image data）
        self.left_time_range = config.left_time_range  # 左侧时间范围（Left time range）
        self.right_time_range = config.right_time_range  # 右侧时间范围（Right time range）
        self.max_len = config.max_len  # 最大长度（Maximum length）
        self.max_cnum = config.max_cnum  # 最大评论数（Maximum number of comments）
        self.is_train = config.is_train  # 是否为训练集（Is it a training set）
        self.set_name = config.set_name  # 数据集名称（Dataset name）

        self.datas = []  # 初始化数据列表（Initialize the data list）

        # 从词汇表中获取特殊符号的ID（Get the IDs of special symbols from the vocabulary）
        self.BOS = config.vocabs["<BOS>"]
        self.EOS = config.vocabs["<EOS>"]
        self.UNK = config.vocabs["<UNK>"]
        self.PAD = config.vocabs["<PAD>"]

        # 检查 PAD 符号的 ID 是否为 0（Check if the ID of the PAD symbol is 0）
        if self.PAD != 0:
            logger("Error! Please set <PAD> id 0 in the dict!", log_level="error")
            sys.exit()

        # 加载数据（Load data）
        self.load_testdata() if self.set_name == "test" else self.load_data()

    @timeit
    def load_data(self):
        """
        加载数据集。
        Load the dataset.
        """
        # 清空已有数据
        # Clear existing data
        self.datas.clear()

        def load_and_check_feature(load_func, feature_type, *args):
            """
            加载特征并进行检查。
            Load the feature and perform checks.
            """
            feature = load_func(*args)
            if feature is None:
                # 如果特征缺失，记录警告日志
                # Log a warning if the feature is missing
                logger(f"Skipping sample due to missing {feature_type} feature.", log_level="warning")
            return feature

        def process_line(line):
            """
            处理单行数据并返回数据样本。
            Process a single line of data and return a data sample.
            """
            try:
                # 尝试解析每一行的JSON数据
                # Try to parse the JSON data for each line
                video_metadata = json.loads(line)
            except json.JSONDecodeError:
                # 如果JSON解析失败，记录警告日志并跳过该行
                # Log a warning and skip the line if JSON parsing fails
                logger("Skipping line due to JSON decode error.", log_level="warning")
                return None

            # 从元数据中提取视频ID和时间
            # Extract video ID and time from metadata
            video_id, video_time = video_metadata["video"], video_metadata["time"]

            # 计算视频特征的时间范围
            # Calculate the time range for video features
            start_time_range = video_time - self.left_time_range
            end_time_range = video_time + self.right_time_range

            # 加载并检查各种特征
            # Load and check various features
            # 准时间及其前后特定时间区间的视频特征。
            # Video features of the baseline time and its specific time intervals before and after.
            video_feature = load_and_check_feature(
                self.load_images, "video", video_id, start_time_range, end_time_range
            )
            # 基准时间及其前后特定时间区间的上下文特征（指定了每条评论的最大长度以及上下文最大评论条数）
            # Contextual features of the baseline time and its specific time intervals before and after (specifying the maximum length of each comment and the maximum number of comments in the context).
            context_feature = load_and_check_feature(
                self.load_comments, "context", video_metadata["context"], start_time_range, end_time_range, video_time
            )
            # 基准时间点的评论特征
            # Comment features at the reference time point
            comment_feature = load_and_check_feature(self.padding, "comment", video_metadata["comment"])

            # 检查所有特征是否都存在
            # Check if all features are present
            if all(v is not None for v in [video_feature, context_feature, comment_feature]):
                # 创建并返回数据样本
                # Create and return the data sample
                return {
                    "video_id": video_id,
                    "video_time": video_time,
                    # "comment": video_metadata["comment"],  # 不确定是否有用，先保留
                    # "context": video_metadata["context"],  # 不确定是否有用，先保留
                    "video_feature": video_feature,
                    "context_feature": context_feature,
                    "comment_feature": comment_feature
                }

        # 使用 ExitStack 管理资源，以便在异常情况下正确关闭文件
        # Use ExitStack to manage resources so that files are properly closed in case of exceptions
        with ExitStack() as stack:
            try:
                # 尝试打开数据文件
                # Try to open the data file
                fin = stack.enter_context(open(self.data_path, "r", encoding="utf-8"))
            except FileNotFoundError:
                # 如果文件不存在，记录错误日志并设置数据为空列表
                # Log an error and set the data to an empty list if the file is not found
                logger("Data file not found.", log_level="error")
                self.datas = []
            else:
                # 使用生成器处理所有行，然后过滤掉 None 值
                # Process all lines using a generator, then filter out None values
                result = list(filter(None, (process_line(line) for line in islice(fin, 1000))))
                self.datas = result

    def load_testdata(self):
        count = 1000
        start_time = time.time()
        self.datas = []
        self.processed_img = {}
        with open(self.data_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                count -= 1
                jterm = json.loads(line)
                video_id = jterm['video']
                video_time = jterm['time']
                sample = {'video_id': video_id,
                          'video_time': video_time,
                          'comment': jterm['comment'],
                          'context': jterm['context']}
                start_time = video_time - self.left_time_range
                end_time = video_time + self.right_time_range

                # format video feature
                video_feature = self.load_imgs(video_id, start_time, end_time)
                if video_feature is None:
                    continue
                sample['video_feature'] = video_feature

                # format ground truth comments
                sample['context_feature'] = self.load_comments(jterm['context'], start_time, end_time, video_time)

                # sample['comment_feature'] = self.padding(jterm['comment'])
                if 'candidate' in jterm:
                    sample['candidate'] = jterm['candidate']
                    sample['candidate_feature'] = [self.padding(c) for c in jterm['candidate']]
                self.datas.append(sample)

        print('Finish loading data ', len(self.datas), ' samples')
        print('Time ', time.time() - start_time)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        logger(f"获取训练特征数据：{index}")
        data = self.datas[index]
        video_data = data["video_feature"]
        sentence_data = data["context_feature"]
        label_data = data["comment_feature"]
        logger("获取训练特征数据完成")
        return video_data, sentence_data, label_data

    def get_data(self, index):
        return self.datas[index]

    def load_images(self, video_id, start_time, end_time):
        """
        加载指定视频ID和时间范围内的图像。Load images for a given video ID and time range.

        参数 (Parameters):
        - video_id (str): 视频的唯一标识符。Unique identifier for the video.
        - start_time (int): 开始时间（包括）。Start time (inclusive).
        - end_time (int): 结束时间（包括）。End time (inclusive).

        返回 (Returns):
        - torch.Tensor: 如果所有图像都存在，则返回图像的张量堆栈。Tensor stack of images if all images are present.
        - None: 如果任何图像不存在。None if any image is missing.
        """

        # 参数有效性检查 (Check for valid input types)
        if not isinstance(video_id, str) or not isinstance(start_time, int) or not isinstance(end_time, int):
            error_message = "Invalid input types."
            logger(error_message, log_level="error")
            raise ValueError(error_message)

        # 检查时间范围有效性 (Check for valid time range)
        if start_time > end_time:
            error_message = "Start time should be less than or equal to end time."
            logger(error_message, log_level="error")
            raise ValueError(error_message)

        # 获取视频图像字典，避免 KeyError (Get the dictionary of video images, avoiding KeyError)
        video_images = self.images.get(video_id, {})

        # 创建时间集合 (Create a set of times)
        times = set(range(start_time, end_time + 1))

        # 一次性检查所有时间帧是否存在 (Check all time frames at once for existence)
        if all(frame_time in video_images for frame_time in times):
            # 预分配内存空间 (Pre-allocate memory space)
            image_shape = video_images[start_time].shape
            image_tensor = torch.empty((len(times), *image_shape), dtype=torch.float32)

            # 填充预分配的张量 (Fill the pre-allocated tensor)
            for i, frame_time in enumerate(times):
                np.copyto(image_tensor[i].numpy(), video_images[frame_time])
            return image_tensor
        else:
            # 记录缺失的时间帧 (Log the missing time frames)
            missing_times = [frame_time for frame_time in times if frame_time not in video_images]
            logger(f"Missing images for Video: {video_id} at times: {missing_times}")
            return None

    def load_comments(self, comments, start_time, end_time, exclude_time):
        """
        加载给定时间范围内的评论，并排除 exclude_time 处的评论。
        Load comments for a given time range and exclude comments at the exclude_time.

        参数 (Parameters):
        - comments_dict (dict): 以时间为索引的评论字典。Dictionary containing comments indexed by time.
        - start_time (int): 开始时间（包括）。Start time (inclusive).
        - end_time (int): 结束时间（包括）。End time (inclusive).
        - exclude_time (int): 要排除的时间（基准真值）。Time to exclude (ground truth).

        返回 (Returns):
        - torch.Tensor: 评论的张量堆栈。Tensor stack of comments.
        """

        def get_comments_for_time(frame_time):
            """
            获取指定时间的评论并进行填充。
            Get comments for a specific time and perform padding.

            参数 (Parameters):
            - frame_time (int): 当前帧的时间。Time of the current frame.

            返回 (Returns):
            - torch.Tensor: 填充后的评论张量。Padded tensor of comments.
            """
            current_comments = comments.get(str(frame_time), [])

            # 获取填充后的评论，直到达到 max_cnum
            # Get padded comments up to max_cnum
            padded_comments = [self.padding(comment) for comment in current_comments[:self.max_cnum]]

            # 填充以确保所有评论列表具有相同的长度
            # Padding to ensure all comment lists have the same length
            padding_count = self.max_cnum - len(padded_comments)
            if padding_count < 0 or not isinstance(padding_count, int):
                raise ValueError("Invalid padding_count value. It should be a non-negative integer.")
            padded_comments.extend([self.padding('')] * padding_count)

            return torch.stack(padded_comments)

        # 使用列表推导式获取评论列表，排除 exclude_time
        # Use list comprehension to get the list of comments, excluding exclude_time
        comments_list = [get_comments_for_time(frame_time) for frame_time in range(start_time, end_time + 1) if
                         frame_time != exclude_time]

        # 返回评论的张量堆栈
        # Return the tensor stack of comments
        return torch.stack(comments_list)

    def padding(self, data):
        """
        对输入数据进行填充和截断。

        参数 (Parameters):
        - data (str): 输入的评论数据。Input comment data.

        返回 (Returns):
        - torch.Tensor: 填充和截断后的张量。Padded and truncated tensor.
        """

        # 分割数据并截断到 max_len - 2
        # Split the data and truncate to max_len - 2
        truncated_data = data.split(' ')[:self.max_len - 2]

        # 将数据转换为词汇表索引
        # Convert data to vocabulary indices
        vocab_indices = [self.vocabs.get(token, self.UNK) for token in truncated_data]

        # 添加开始和结束标记
        # Add start and end tokens
        vocab_indices = [self.BOS] + vocab_indices + [self.EOS]

        # 计算填充数量并进行填充
        # Calculate the padding amount and pad
        padding_count = self.max_len - len(vocab_indices)
        padded_data = vocab_indices + [0] * padding_count

        return torch.LongTensor(padded_data)


class DataHandler(object):

    def __init__(self, args, images, vocabs, reverse_vocabs):
        self.args = args
        self.images = images
        self.vocabs = vocabs
        self.reverse_vocabs = reverse_vocabs

    def get_dataset(self, data_path, images, is_train, set_name):
        """
        获取数据集对象（Get the dataset object）

        参数 (Parameters):
        - data_path (str): 数据的路径（Path to the data）
        - images (dict): 图像数据（Image data）
        - is_train (bool): 是否用于训练（Whether it's for training）
        - set_name (str): 数据集名称（Name of the dataset）

        返回 (Returns):
        - dataset.Dataset: 数据集对象（Dataset object）
        """
        config = DatasetConfig(
            data_path=data_path,                    # 数据的路径（Path to the data）
            vocabs=self.vocabs,                     # 词汇表（Vocabulary list）
            reverse_vocabs=self.reverse_vocabs,     # 反向词汇表（Reverse vocabulary list）
            images=images,                          # 图像数据（Image data）
            left_time_range=self.args.time_range,   # 左侧时间范围（Left time range）
            right_time_range=self.args.time_range,  # 右侧时间范围（Right time range）
            max_len=self.args.max_len,              # 文本最大长度（Maximum text length）
            max_cnum=self.args.max_cnum,            # 每秒最大评论数（Maximum number of comments per second）
            is_train=is_train,                      # 是否用于训练（Whether it's for training）
            set_name=set_name                       # 数据集名称（Name of the dataset）
        )

        return Dataset(config)

    def get_dataloader(self, data, batch_size, is_train, num_workers=4):
        return TorchDataLoader(
            dataset=data,             # 要加载的数据集（Dataset to load）
            batch_size=batch_size,    # 每个批次的大小（Size of each batch）
            shuffle=is_train,         # 是否打乱数据（Whether to shuffle the data）
            num_workers=num_workers,  # 用于数据加载的子进程数（Number of subprocesses to use for data loading）
            pin_memory=True           # 将数据存储在固定内存中（这通常会加速数据传输到 GPU）（Store data in pinned memory (this usually speeds up data transfer to the GPU)）
        )


# def compare_dicts(dict1, dict2):
#     for key in dict1.keys():
#         if key not in dict2:
#             return False
#         if isinstance(dict1[key], torch.Tensor) and isinstance(dict2[key], torch.Tensor):
#             if not torch.equal(dict1[key], dict2[key]):
#                 return False
#         elif dict1[key] != dict2[key]:
#             return False
#     return True
#
# is_equal = all(compare_dicts(d1, d2) for d1, d2 in zip(result, result_for))
#
# if is_equal:
#     logger("All methods produced the same results.")
# else:
#     logger("Methods produced different results.")