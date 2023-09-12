import h5py
import json
import logging
import numpy as np

from typing import Tuple, Dict, Generator, Iterable


class DataLoader(object):
    def __init__(self, image_path=None, vocabulary_path=None):
        self.image_data_path = image_path
        self.vocabulary_data_path = vocabulary_path

    def load_vocabs(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        加载词汇表和反向词汇表（Load vocabulary and reverse vocabulary）

        参数 (Parameters):
        - data_path (str): 存储词汇表 JSON 文件的目录路径（The directory path where the vocabulary JSON file is stored）

        返回 (Returns):
        - vocabs (dict): 单词到 ID 的映射（Mapping from word to ID）
        - rev_vocabs (dict): ID 到单词的映射（Mapping from ID to word）
        """

        # 从JSON文件中加载词汇表 (Load the vocabulary from the JSON file)
        vocabs = json.load(open(self.vocabulary_data_path, "r", encoding="utf-8"))["word2id"]

        # 创建反向词汇表，即从ID到单词的映射 (Create a reverse vocabulary, i.e., a mapping from ID to word)
        reverse_vocabs = {vocabs[k]: k for k in vocabs}

        # 打印加载的词汇表的长度 (Print the length of the loaded vocabulary)
        print("Load vocabs ", len(vocabs))

        # 返回正向和反向词汇表 (Return both the forward and reverse vocabularies)
        return vocabs, reverse_vocabs

    def load_images(self) -> Generator[Dict[str, Dict[int, np.ndarray]], None, None]:
        """
        从HDF5文件中加载图像，并将它们作为嵌套字典返回（Load images from an HDF5 file and return them as a nested dictionary）

        参数 (Parameters):
        - data_path (str): 包含HDF5文件的目录的路径 （The path to the directory containing the HDF5 file）

        返回 (Returns):
        - images (dict): 包含已加载图像的嵌套字典（A nested dictionary containing the loaded images）
        """

        def load_group_data(group: h5py.Group) -> Dict[int, np.ndarray]:
            """
            从单个HDF5组中加载数据（Load data from a single HDF5 group）

            参数 (Parameters):
            - group (HDF5 group object): HDF5组对象（HDF5 group object）

            返回 (Returns):
            - Dictionary containing the loaded data from the group (dict): 包含从组中加载的数据的字典（Dictionary containing the loaded data from the group）
            """
            sorted_keys = sorted(group.keys(), key=int)
            return dict(map(lambda k: (int(k), group[k][:]), sorted_keys))

        def image_generator(hf: h5py.File) -> Iterable[Dict[str, Dict[int, np.ndarray]]]:
            keys = list(hf.keys())
            return ({key: load_group_data(hf[key])} for key in keys)

        if not self.image_data_path.exists():
            logging.error(f"The HDF5 file does not exist at {self.image_data_path}")
            return

        try:
            with h5py.File(self.image_data_path, "r") as hf:
                yield from image_generator(hf)
        except (FileNotFoundError, OSError) as e:
            logging.error(f"An error occurred while reading the HDF5 file at {self.vocabulary_data_path}: {e}")
        except Exception as e:
            logging.error(f"An unknown error occurred while reading the HDF5 file: {e}")