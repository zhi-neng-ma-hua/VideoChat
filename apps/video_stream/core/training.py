import os
import time
import torch

from tqdm import tqdm


from VideoChatGPT.settings import CUDA_DEVICES

from data_loader import DataLoader
from data_handler import DataHandler

from utils.argument_parser import ArgumentParser
from utils.decorators import timeit
from utils.logger_config import logger, setup_logging


class TrainingHelper(object):
    def __init__(self, data_handler, args):
        """
        构造函数（Constructor）
        """
        # 数据处理对象（Data handler object）
        self.data_handler = data_handler
        # 参数（Arguments）
        self.args = args

    @timeit
    def load_data(self, set_name):
        """
        加载数据（Load data）
        """
        return self.data_handler.get_dataset(
            data_path=os.path.join(self.args.data_path, f"{set_name}.json"),
            images=self.data_handler.images,
            is_train=(set_name == "train"),
            set_name=set_name
        )

    def initialize_model(self):
        """
        初始化模型（Initialize model）
        """
        config = modules.ModelConfig(
            embedding_dim=self.args.n_emb,             # 嵌入层大小（Size of the embedding layer）
            hidden_dim=self.args.n_hidden,             # 隐藏层大小（Size of the hidden layer）
            num_heads=self.args.n_head,                # 多头注意力机制的头数（Number of heads in multi-head attention mechanism）
            num_blocks=self.args.n_block,              # 模型中块的数量（Number of blocks in the model）
            max_sequence_length=self.args.max_len,     # 序列的最大长度（Maximum length of the sequence）
            dropout=self.args.dropout,                 # dropout 概率（Dropout probability）
            vocab_size=len(self.data_handler.vocabs),  # 词汇表大小（Vocabulary size）
            left_time_range=self.args.time_range,      # 左侧时间范围（Left time range）
            right_time_range=self.args.time_range      # 右侧时间范围（Right time range）
        )

        model = modules.Model(config)
        return model

    def check_early_stopping(self, current_score, best_score, early_stop_counter):
        """
        检查是否需要早停（Check for early stopping）
        """
        # 是否保存最佳模型的标志（Flag to indicate if the best model should be saved）
        save_best_model = False
        # 是否停止训练的标志（Flag to indicate if training should be stopped）
        stop_training = False

        if current_score > best_score:
            best_score = current_score
            early_stop_counter = 0
            save_best_model = True
        else:
            early_stop_counter += 1

            # 如果早停计数器达到阈值，停止训练（If the early stop counter reaches the threshold, stop training）
            if early_stop_counter == self.args.early_stop:
                stop_training = True

        return best_score, early_stop_counter, save_best_model, stop_training

    @timeit
    def create_dataloader(self, data, is_train=True):
        """
        创建 DataLoader（Create DataLoader）
        """
        return self.data_handler.get_dataloader(data, self.args.batch_size, is_train)


class MMIGModel(object):
    def __init__(self, args, data_handler):
        """
        构造函数（Constructor）
        """
        self.args = args
        self.data_handler = data_handler
        # self.logger = self.initialize_logger()

    @staticmethod
    def save_model(path, model):
        """
        保存模型（Save model）
        """
        torch.save(model.state_dict(), path)
        # # 保存整个模型（Save the entire model）
        # torch.save(model, path)

    def transform_ids_to_word(self, ids):
        """
        将ID转换为单词（Transform IDs to words）
        """
        # # 初始化一个空列表用于存储句子（Initialize an empty list to store sentences）
        # sentences = []
        #
        # # 遍历输入列表 'ids' 中的每一个词ID（Loop through each word ID in the input list 'ids'）
        # for word_id in ids:
        #     # 跳过句子开始标记 '<BOS>'（Skip the beginning-of-sentence token）
        #     if word_id == self.data_loader.vocabs['<BOS>']: continue
        #     # 如果遇到句子结束标记 '<EOS>'，则跳出循环（Break the loop if end-of-sentence token is encountered）
        #     if word_id == self.data_loader.vocabs['<EOS>']: break
        #     # 将与词ID对应的词追加到 'sentences' 列表中（Append the word corresponding to the word ID to the 'sentences' list）
        #     sentences.append(self.data_loader.reverse_vocabs[word_id])
        #
        # # 返回句子列表（Return the list of sentences）
        # return sentences
        return [self.data_handler.reverse_vocabs[word_id] for word_id in ids if word_id not in [self.data_handler.vocabs["<BOS>"], self.data_handler.vocabs["<EOS>"]]]

        # sentences = [self.data_loader.reverse_vocabs[word_id] for word_id in ids if
        #          word_id != self.data_loader.vocabs['<BOS>'] and word_id != self.data_loader.vocabs['<EOS>']]
        # return sentences

    def train_model(self):
        """
        训练模型的主函数（Main function for training the model）
        """

        logger("Training started.")

        training_helper = TrainingHelper(self.data_handler, self.args)

        # 加载训练数据集（Load the training dataset）
        logger("Loading training dataset...")
        train_dataset = training_helper.load_data("train")

        # 加载训练和验证数据集（Load the validation dataset）
        logger("Loading validation dataset...")
        validation_dataset = training_helper.load_data("dev")
        logger("Validation datasets loaded.")

        # 为训练集创建一个 DataLoader（Create a DataLoader for the training set）
        logger("Creating DataLoader for training set...")
        train_batches = training_helper.create_dataloader(train_dataset, is_train=True)
        logger("DataLoader created.")

        # 使用指定参数初始化模型（Initialize the model with specified parameters）
        logger("Initializing the model...")
        model = training_helper.initialize_model()
        logger("Model initialized.")

        # 如果提供了恢复路径，则加载预训练模型（Load a pre-trained model if a restore path is provided）
        if self.args.restore:
            logger(f"Loading pre-trained model from {self.args.restore}...")
            model.load_state_dict(torch.load(self.args.restore))
            logger("Pre-trained model loaded.")

        # 将模型移至 GPU（Move the model to GPU）
        if torch.cuda.is_available():
            logger("Moving the model to GPU...")
            model = model.cuda()
            # model = torch.nn.DataParallel(model)
            logger("Model moved to GPU.")

        # 初始化 Adam 优化器（Initialize the Adam optimizer）
        logger("Initializing the optimizer...")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        logger("Optimizer initialized.")

        # 初始化学习率调度器（Initialize learning rate scheduler）
        logger("Initializing learning rate scheduler...")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
        logger(" ||--Scheduler initialized.")

        best_score, early_stop_counter = -100000, 0

        # 开始训练循环（Start the training loop）
        logger(" ||--Starting the training loop...")
        for epoch in range(self.args.epoch):
            model.train()
            total_loss, start_time = 0, time.perf_counter()

            # 从 DataLoader 中迭代每个批次（Iterate over each batch from the DataLoader）
            for batch_idx, (video_data, sentence_data, label_data) in enumerate(tqdm(train_batches), 1):
                # 清零梯度（Zero the gradients）
                optimizer.zero_grad()

                # video_data: 视频特征，sentence_data: 周围的评论，label_data: 真实值（video_data: video feature, sentence_data: Surrounding comments, label_data: Ground truth）
                # V = Variable(V).cuda()
                # S = Variable(S).cuda()
                # Y = Variable(Y).cuda()
                # video_data, sentence_data, label_data = map(Variable().cuda, (video_data, sentence_data, label_data))
                video_data, sentence_data, label_data = map(lambda x: x.cuda(), [video_data, sentence_data, label_data])

                # 前向传播（Forward pass）
                batch_loss = model(video_data, sentence_data, label_data)
                # loss = torch.sum(batch_loss)
                loss = torch.sum(batch_loss) / len(video_data)

                # 反向传播和优化（Backward pass and optimization）
                loss.backward()
                optimizer.step()

                # 更新损失指标（Update the loss metrics）
                total_loss += torch.mean(batch_loss).item()

                # 释放不再需要的变量以节省内存（Release variables to save memory）
                del video_data, sentence_data, label_data, batch_loss
                torch.cuda.empty_cache()

            # 报告这个周期的损失（Report the loss for this epoch）
            avg_loss = total_loss / batch_idx if batch_idx > 0 else 0
            logger(f" ||--Epoch {epoch}, Avg Loss: {avg_loss}, Time: {time.perf_counter() - start_time:.2f}s")

            # 在验证集上评估模型（Evaluate the model on the validation set）
            logger(" ||--Evaluating the model on the validation set...")
            with torch.no_grad():  # 禁用梯度计算以加速评估（Disable gradient computation for faster evaluation）
                current_score = self.evaluate(model, validation_dataset)
            logger(f" ||--Evaluation score: {current_score}")

            # 检查是否需要早停（Check for early stopping）
            best_score, early_stop_counter, save_best_model, stop_training = training_helper.check_early_stopping(
                current_score, best_score, early_stop_counter
            )

            # 如果需要保存最佳模型，执行保存操作（If the best model needs to be saved, perform the save operation）
            if save_best_model:
                self.save_model(os.path.join(self.args.out_path, "best_checkpoint.pt"), model)
                logger(f" ||--New best score: {best_score}. Saving best model.")
                # 更新学习率（Update learning rate）
                scheduler.step(current_score)

            # 如果需要早停，跳出循环（If early stopping is needed, break the loop）
            if stop_training:
                self.logger.warning(" ||--Early stopping triggered.")
                break

    def evaluate(self, model, validation_set):
        """
        在验证集上评估模型（Evaluate the model on the validation set）
        """
        # 打印开始评估的消息（Print start message for evaluation）
        print('Start Evaluation ... ')

        # 记录评估开始的时间（Record the start time for evaluation）
        start_time = time.perf_counter()

        # 将模型设置为评估模式（Set the model to evaluation mode）
        model.eval()

        # 获取验证数据加载器（Get the validation data loader）
        validation_batches = self.data_handler.get_dataloader(validation_set, self.args.batch_size, is_train=False)

        # 初始化损失和批次计数的变量（Initialize variables for loss and batch count）
        total_loss = 0

        # 禁用梯度计算（Disable gradient calculation）
        with torch.no_grad():
            # 从 DataLoader 中迭代每个批次（Iterate over each batch from the DataLoader）
            for batch_idx, (video_data, sentence_data, label_data) in enumerate(tqdm(validation_batches), 1):
                # 将数据移动到 GPU（Move data to GPU）
                video_data, sentence_data, label_data = map(lambda x: x.cuda(), [video_data, sentence_data, label_data])

                # 计算并累积损失（Calculate and accumulate the loss）
                batch_loss = model(video_data, sentence_data, label_data)
                total_loss += torch.mean(batch_loss).item()

        # 计算平均损失（Calculate the average loss）
        avg_loss = total_loss / batch_idx if batch_idx > 0 else 0  # 避免除以零（Avoid division by zero）

        # 计算并打印评估所需的时间（Calculate and print the time taken for evaluation）
        elapsed_time = time.perf_counter() - start_time
        print(f'Loss: {avg_loss}')
        print(f"Evaluating time: {elapsed_time} seconds")

        # 返回负损失（用于优化）（Return the negative loss (used for optimization)）
        return -avg_loss

def run():
    # 设置可见的 CUDA 设备（Set the visible CUDA devices for this script）
    logger("Setting CUDA devices...")
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES
    logger("CUDA devices set.")

    logger("Parsing command-line arguments...")
    args = ArgumentParser().args
    logger("Command-line arguments parsed.")

    # 设置随机种子（Set random seed）
    logger("Setting random seed...")
    torch.manual_seed(116)
    if torch.cuda.is_available(): torch.cuda.manual_seed(116)
    logger("Random seed set.")

    data_loader = DataLoader(image_path=args.image_data_path, vocabulary_path=args.vocabulary_data_path)
    # 加载图像（Load images）
    logger("Loading images...")
    images = {}
    for image_batch in data_loader.load_images(): images.update(image_batch)
    logger("Images loaded.")

    # 加载词汇（Load vocabularies）
    logger("Loading vocabularies...")
    vocabs, reverse_vocabs = data_loader.load_vocabs()
    logger("Vocabularies loaded.")

    # 初始化数据加载器和模型（Initialize data loader and model）
    logger("Initializing data loader and model...")
    data_loader = DataHandler(args, images, vocabs, reverse_vocabs)
    model = MMIGModel(args, data_loader)
    logger("Data loader and model initialized.")

    # 根据模式进行训练或测试（Train or test based on the mode）
    logger("Determining mode for train/test...")
    if args.mode == 'train':
        logger("Training mode selected.")
        model.train_model()
    elif args.mode == 'test':
        logger("Testing mode selected.")
        model.test_model()
    else:
        logger("Invalid mode. Choose between 'train' and 'test'.", log_level="warning")


if __name__ == '__main__':
    # 初始化日志设置（Initialize logging settings）
    setup_logging()

    run()