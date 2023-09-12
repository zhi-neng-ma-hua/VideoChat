import math
import numpy as np
import torch

from torch import nn
from torch.autograd import Variable


class ModelConfig:
    def __init__(self, embedding_dim, hidden_dim, num_heads, num_blocks, max_sequence_length,
                 dropout, vocab_size, left_time_range, right_time_range, image_dim=2048):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.left_time_range = left_time_range
        self.right_time_range = right_time_range
        self.image_dim = image_dim


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Model, self).__init__()

        # todo 设置参数（Set parameters）
        self.embedding_dim = config.embedding_dim              # 词嵌入维度（Word embedding dimension）
        self.hidden_dim = config.hidden_dim                    # 隐藏层维度（Hidden layer dimension）
        self.num_heads = config.num_heads                      # 多头注意力的头数（Number of heads in multi-head attention）
        self.num_blocks = config.num_blocks                    # Transformer块的数量（Number of Transformer blocks）
        self.vocab_size = config.vocab_size                    # vocab_size: 词汇表大小（Vocabulary size）
        self.max_sequence_length = config.max_sequence_length  # 序列的最大长度（Maximum length of the sequence）
        self.dropout = config.dropout                          # dropout率（Dropout rate）
        self.image_dim = config.image_dim                      # 图像维度（Image dimension）
        self.left_time_range = config.left_time_range          # 左侧时间范围（Left time range）
        self.right_time_range = config.right_time_range        # 右侧时间范围（Right time range）

        # todo 视频编码器（Video encoder）
        # 全局视频LSTM编码器（Global video LSTM encoder）
        self.global_video_encoder = VideoLSTM(self.image_dim, self.hidden_dim, self.dropout)
        # 局部视频注意力编码器（Local video attention encoder）
        self.local_video_encoder = VideoAttention(self.image_dim, self.hidden_dim, self.num_heads, self.embedding_dim, self.dropout)

        # todo 文本编码器（Text encoder）
        # 词嵌入层（Word embedding layer）
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        # 文本LSTM编码器（Text LSTM encoder）
        self.sentence_encoder = TextLSTM(self.embedding_dim, self.hidden_dim, self.dropout)

        # todo 时间预测器（Temporal predictor）
        # 多模态注意力层（Multi-modal attention layer）
        self.multi_modal_attention = MultiModalAttention(self.embedding_dim, self.dropout)
        # 时间预测线性层（Temporal prediction linear layer）
        self.temporal_predictor = nn.Linear(2 * self.hidden_dim, 2)

        # todo 评论解码器（Comments decoder）
        # 评论解码器（Comment decoder）
        self.comment_decoder = CommentDecoder(self.embedding_dim, self.hidden_dim, self.num_heads, self.num_blocks, self.dropout)
        # 输出层（Output layer）
        self.output_layer = nn.Linear(self.embedding_dim, self.vocab_size)

        # todo 损失函数（Loss functions）
        # 生成损失（Generation loss）
        self.generation_loss = nn.CrossEntropyLoss(reduce=False, size_average=False, ignore_index=0)
        # 分类损失（Classification loss）
        self.classification_loss = nn.CrossEntropyLoss()
        
    def encode_image(self, V):
        img_global = self.global_video_encoder(V)
        img_local = self.local_video_encoder(V)
        return img_global, img_local
    
    def encode_text(self, T):
        T_embs = self.word_embedding(T)
        T_encode = self.sentence_encoder(T_embs)
        return T_encode
    
    def decode(self, Y, text_v, mask):
        embs = self.word_embedding(Y)
        out = self.decoder(embs, text_v, mask)
        out = self.output_layer(out)
        return out
        
    def forward(self, V, T, Y):
        # encode image
        img_global, img_local = self.encode_image(V)

        # encode surrounding comments
        text = self.encode_text(T)

        # classification
        loss_c, text_v = self.classifier(img_global, img_local, text)
        mask = Variable(subsequent_mask(Y.size(0), Y.size(1)-1), requires_grad=False).cuda()
        
        # decode
        outs = self.decode(Y[:,:-1], text_v, mask)

        # compute loss
        Y = Y.t()
        outs = outs.transpose(0, 1)
        loss_g = self.generation_loss(outs.contiguous().view(-1, self.vocab_size), Y[1:].contiguous().view(-1))
        loss = 0.7*torch.mean(loss_g) + 0.3*loss_c
        # print('Total loss', loss)
        return loss
        
    def classifier(self, img_global, img_local, text):
        '''
            Whether comments appears time stamp t
        '''
        batch_size = img_global.size(0)
        IG = (img_global.unsqueeze(1)).repeat(1, text.size(1), 1)
        c = self.multi_modal_attention(IG.view(-1, 1, IG.size(-1)), text.view(-1, text.size(-2), text.size(-1)))
        text_topic = c.view(batch_size, -1, c.size(-1))
        #print('Text topic ', text_topic.size())
        vt_feature = torch.cat((text_topic, img_global.unsqueeze(1), img_local.unsqueeze(1)), dim=1)
        #print('Vt_feature ', vt_feature.size())
        
        # classification
        vt = torch.cat((text_topic, (img_local.unsqueeze(1)).repeat(1, text_topic.size(1), 1)), dim=-1)
        vt_predict = self.temporal_predictor(vt)
        before = Variable(torch.LongTensor(batch_size, self.left_time_range).fill_(1)).cuda()
        after = Variable(torch.LongTensor(batch_size, self.right_range).fill_(0)).cuda()
        labels = torch.cat((before, after), dim=1)
        loss = self.classification_loss(vt_predict.view(-1, 2), labels.view(-1))

        return loss, vt_feature
    
    def ranking(self, V, T, C):
        nums = len(C)
        img_global, img_local = self.encode_image(V.unsqueeze(0))
        text = self.encode_text(T.unsqueeze(0))
        loss_c, text_v = self.classifier(img_global, img_local, text)
        text_v = text_v.repeat(nums, 1, 1)
        mask = Variable(subsequent_mask(C.size(0), C.size(1) - 1), requires_grad=False).cuda()
        outs = self.decode(C[:, :-1], text_v,  mask)

        C = C.t()
        outs = outs.transpose(0, 1)

        loss = self.generation_loss(outs.contiguous().view(-1, self.vocab_size), C[1:].contiguous().view(-1))

        loss = loss.view(-1, nums).sum(0)
        return torch.sort(loss, dim=0, descending=False)[1]
        
        
    
    def generate(self, V, T, BOS_token, EOS_token, beam_size):
        img_global, img_local = self.encode_image(V.unsqueeze(0))
        text = self.encode_text(T.unsqueeze(0))
    
        loss_c, text_v = self.classifier(img_global, img_local, text)
        comments = self.beam_search(text_v, BOS_token, EOS_token, beam_size)   
        return comments
    
    def beam_search(self, text_v, BOS_token, EOS_token, beam_size):
        LENGTH_NORM = True
        batch_size = text_v.size(0)
        startTokenArray = Variable(torch.LongTensor(batch_size, 1).fill_(BOS_token)).cuda()
        #print('Start matrix ', startTokenArray.size())

        backVector = torch.LongTensor(beam_size).cuda()
        torch.arange(0, beam_size, out=backVector)
        backVector = backVector.unsqueeze(0).repeat(batch_size, 1)
        backVector = Variable(backVector)
        #print('Back matrix ', backVector.size())

        tokenArange = torch.LongTensor(self.vocab_size).cuda()
        torch.arange(0, self.vocab_size, out=tokenArange)
        tokenArange = Variable(tokenArange)
        #print('Token matrix ', tokenArange.size())

        beamTokensTable = torch.LongTensor(batch_size, beam_size, self.max_sequence_length).fill_(EOS_token)
        beamTokensTable = Variable(beamTokensTable.cuda())
        #print('beam Table ', beamTokensTable.size())

        backIndices = torch.LongTensor(batch_size, beam_size, self.max_sequence_length).fill_(-1)
        backIndices = Variable(backIndices.cuda())
        #print('Back Indices ', backIndices.size())

        aliveVector = beamTokensTable[:, :, 0].eq(EOS_token).unsqueeze(2)
        #print('AliveVector ', aliveVector.size())

        for i in range(self.max_sequence_length-1):
            if i  == 0:
                Cap = startTokenArray
                mask = Variable(subsequent_mask(Cap.size(0), Cap.size(1))).cuda()
                #print('Mask ', mask.size())
                out = self.decode(Cap, text_v, mask)
                #print('Out ', out.size())
                probs = out[:, -1]
                topProbs, topIdx = probs.topk(beam_size, dim=1)
                beamTokensTable[:, :, 0] = topIdx.data
                ProbSums = topProbs
            else:
                Cap = beamTokensTable[:, :, :i].squeeze(0)
                mask = Variable(subsequent_mask(Cap.size(0), Cap.size(1))).cuda()
                out = self.decode(Cap, text_v.repeat(beam_size, 1, 1), mask)
                probCurrent = out[:, -1,:].view(batch_size, beam_size, self.vocab_size)

                if LENGTH_NORM:
                    probs = probCurrent * (aliveVector.float() / (i+1))
                    coeff_ = aliveVector.eq(0).float() + (aliveVector.float() * i / (i+1))
                    probs += ProbSums.unsqueeze(2) * coeff_
                else:
                    probs = probCurrent * (aliveVector.float())
                    probs += ProbSums.unsqueeze(2)

                mask_ = aliveVector.eq(0).repeat(1, 1, self.vocab_size)
                mask_[:, :, 0] = 0
                minus_infinity_ = torch.min(probs).item()

                probs.data.masked_fill_(mask_.data, minus_infinity_)
                probs = probs.view(batch_size, -1)

                tokensArray = tokenArange.unsqueeze(0).unsqueeze(0).repeat(batch_size, beam_size, 1)
                tokensArray.masked_fill_(aliveVector.eq(0), 2)
                tokensArray = tokensArray.view(batch_size, -1)
                backIndexArray = backVector.unsqueeze(2).repeat(1, 1, self.vocab_size).view(batch_size, -1)

                topProbs, topIdx = probs.topk(beam_size, dim=1)
                ProbSums = topProbs
                beamTokensTable[:, :, i] = tokensArray.gather(1, topIdx)
                backIndices[:, :, i] = backIndexArray.gather(1, topIdx)

            aliveVector = beamTokensTable[:, :, i:i + 1].ne(2)
            aliveBeams = aliveVector.data.long().sum()
            finalLen = i
            if aliveBeams == 0:
                break

        beamTokensTable = beamTokensTable.data
        backIndices = backIndices.data

        RECOVER_TOP_BEAM_ONLY = True
        tokenIdx = finalLen
        backID = backIndices[:, :, tokenIdx]
        tokens = []
        while tokenIdx >= 0:
            tokens.append(beamTokensTable[:, :, tokenIdx].gather(1, backID).unsqueeze(2))
            backID = backIndices[:, :, tokenIdx].gather(1, backID)
            tokenIdx = tokenIdx - 1

        tokens.append(startTokenArray.unsqueeze(2).repeat(1, beam_size, 1).data)
        tokens.reverse()
        tokens = torch.cat(tokens, 2)
        seqLen = tokens.ne(2).long().sum(dim=2)

        if RECOVER_TOP_BEAM_ONLY:
            tokens = tokens[:, 0]
            seqLen = seqLen[:, 0]
            
        return Variable(tokens)        


class VideoEncoder(nn.Module):
    def __init__(self, dim, dim_ff, n_head, n_block, dropout):
        super(VideoEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(dim, dim_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class VideoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers=2):
        super(VideoLSTM, self).__init__()

        # 使用构造函数参数初始化类属性
        # Initialize class attributes using constructor parameters
        self.embedding_dim = input_size
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # 在LSTM构造函数中添加dropout参数
        # Add dropout parameter directly in the LSTM constructor
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # dropout=dropout,  先注释掉，调试
            batch_first=True
        )

    def forward(self, input_data):
        # 初始化隐藏状态和细胞状态（Initialize hidden and cell states）
        # 使用.to(input_data.device)确保张量在相同的设备上（Use .to(input_data.device) to ensure tensors are on the same device）
        hidden_state = torch.zeros(self.num_layers, input_data.size(0), self.hidden_dim).to(input_data.device)
        cell_state = torch.zeros(self.num_layers, input_data.size(0), self.hidden_dim).to(input_data.device)

        # hidden_state = Variable(torch.zeros(self.num_layers, input_data.size(0), self.hidden_dim).cuda())
        # cell_state = Variable(torch.zeros(self.num_layers, input_data.size(0), self.hidden_dim).cuda())

        # LSTM前向传播
        # LSTM Forward Propagation
        lstm_output, _ = self.lstm(input_data, (hidden_state, cell_state))

        # 使用切片获取最后一个时间步的输出
        # Use slicing to get the output of the last time step
        output = lstm_output[:, -1, :]

        return output


class VideoAttention(nn.Module):
    def __init__(self, input_dim: int, feed_forward_dim: int, num_heads: int, output_dim: int, dropout: float, num_blocks: int = 1):
        super(VideoAttention, self).__init__()

        # 使用 nn.Sequential 初始化多个注意力块，以便于顺序执行
        # Initialize multiple attention blocks using nn.Sequential for sequential execution
        # self.layers = nn.Sequential(*[AttnBlock(input_dim, feed_forward_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.layers = nn.ModuleList([AttnBlock(input_dim, feed_forward_dim, num_heads, dropout) for _ in range(num_blocks)])

        # 定义一个线性层，用于将注意力层的输出转换为期望的输出维度
        # Define a linear layer to transform the output of the attention layers to the desired output dimension
        self.linear = nn.Linear(input_dim, output_dim)

        # 添加层归一化以稳定模型训练
        # Add layer normalization to stabilize model training
        self.norm = LayerNorm(output_dim)

    def forward(self, input_data):
        # 计算输入数据的中间时间步，用于注意力计算
        # Calculate the middle time step of the input data for attention computation
        mid_time_step = input_data.size(1) // 2
        mid_step_image = input_data[:, mid_time_step, :]

        # 将中间时间步的数据和整个输入数据一起通过多个注意力层
        # Pass the middle time step data along with the entire input data through multiple attention layers
        # for layer in self.layers:
        #     x = layer(img_t.unsqueeze(1), x)
        input_data = self.layers(mid_step_image.unsqueeze(1), input_data)

        # 将注意力层的输出通过一个线性层
        # Pass the output of the attention layers through a linear layer
        # x = self.linear(x.squeeze(1))
        input_data = self.linear(input_data)

        # 对线性层的输出进行层归一化，以得到最终输出
        # Perform layer normalization on the output of the linear layer to get the final output
        return self.norm(input_data)


class TopicAttention(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout, n_block=1):
        super(TopicAttention, self).__init__()
        self.layers = nn.ModuleList([AttnBlock(dim, dim_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(dim)

    def forward(self, x, m):
        for layer in self.layers:
            x = layer(x, m)
        return x


class TextAttention(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout, n_block=4):
        super(TextAttention, self).__init__()
        self.layers = nn.ModuleList([AttnBlock(dim, dim_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, x)
        return x


class MultiModalAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super(MultiModalAttention, self).__init__()

        # 优化点1：添加了一个线性层用于查询（q）的变换，以增加模型的表达能力
        # Optimization Point 1: Added a linear layer for transforming the query (q) to enhance the model's expressiveness
        # self.q_linear = nn.Linear(input_size, input_size)

        # 层归一化（Layer normalization）
        self.norm = LayerNorm(input_size)

        # Dropout 层（Dropout layer）
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, v):
        # 优化点2：通过线性层对查询（q）进行变换
        # Optimization Point 2: Transform the query (q) through the linear layer
        # q = self.q_linear(q)

        # 计算注意力权重（Calculate attention weights）
        weights = torch.bmm(q, v.transpose(1, 2))

        # 使用 softmax 函数进行归一化（Normalize using the softmax function）
        attn_weights = nn.functional.softmax(weights.squeeze(1), dim=1)

        # 应用注意力权重到值（v）上（Apply the attention weights to the values (v)）
        output = torch.bmm(attn_weights.unsqueeze(1), v)

        # 优化点3：添加 dropout 和层归一化
        # Optimization Point 3: Add dropout and layer normalization
        # output = self.dropout(output)
        # output = self.norm(output)

        return output.squeeze(1)


class AttnBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout):
        super(AttnBlock, self).__init__()
        self.attn = MultiHeadAttention(dim, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(dim, dropout) for _ in range(2)])

    def forward(self, x, m):
        x = self.sublayer[0](x, lambda x: self.attn(x, m, m))
        return self.sublayer[1](x, self.feed_forward)


class TextLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers=2):
        super(TextLSTM, self).__init__()

        # 词嵌入维度（Word embedding dimension）
        self.embedding_dim = input_size
        # 隐藏层维度（Hidden layer dimension）
        self.hidden_dim = hidden_size
        # LSTM层数（Number of LSTM layers）
        self.num_layers = num_layers
        # dropout率（Dropout rate）
        self.dropout = dropout

        # 初始化 LSTM 层（Initialize LSTM layer）
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, input_data):
        # 获取批次大小和时间步长（Get batch size and time steps）
        batch_size, time_steps = input_data.size(0), input_data.size(1)

        input_data = input_data.view(-1, input_data.size(-2), input_data.size(-1))

        # 初始化隐藏状态和细胞状态（Initialize hidden and cell states）
        # 使用.to(input_data.device)确保张量在相同的设备上（Use .to(input_data.device) to ensure tensors are on the same device）
        hidden_state = torch.zeros(self.num_layers, input_data.size(0), self.hidden_dim).to(input_data.device)
        cell_state = torch.zeros(self.num_layers, input_data.size(0), self.hidden_dim).to(input_data.device)

        # h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda())
        # c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda())

        # LSTM 前向传播（LSTM forward pass）
        lstm_output, _ = self.lstm(input_data, (hidden_state, cell_state))

        # 获取最后一个时间步的输出（Get the output of the last time step）
        output = lstm_output[:, -1, :]

        # 调整输出形状以匹配原始输入数据的批次和时间步长（Reshape the output to match the original input data's batch and time steps）
        result = output.view(batch_size, time_steps, -1, output.size(-1))

        return result


class CommentDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_blocks, dropout):
        super(CommentDecoder, self).__init__()

        # 优化点1：使用 nn.Sequential 代替 nn.ModuleList，以简化前向传播的代码
        # Optimization Point 1: Use nn.Sequential instead of nn.ModuleList to simplify the forward pass code
        # self.layers = nn.Sequential(*[DecoderBlock(input_size, hidden_size, num_heads, dropout) for _ in range(num_blocks)])
        self.layers = nn.ModuleList([DecoderBlock(input_size, hidden_size, num_heads, dropout) for _ in range(num_blocks)])

        # 层归一化（Layer normalization）
        self.norm = LayerNorm(input_size)

    def forward(self, input_sequence, target_values, attention_mask):
        # 优化点2：直接使用 nn.Sequential 对象，无需循环
        # Optimization Point 2: Directly use the nn.Sequential object, eliminating the need for a loop
        output_sequence = self.layers(input_sequence, target_values, attention_mask)

        # 应用层归一化（Apply layer normalization）
        return self.norm(output_sequence)

        # for layer in self.layers:
        #     x = layer(x, tv, mask)
        #
        # return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(dim, n_head, dropout)
        self.attn = MultiHeadAttention(dim, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(dim, dropout) for _ in range(3)])

    def forward(self, x, tv, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, lambda x: self.attn(x, tv, tv))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_head, dropout):
        super(MultiHeadAttention, self).__init__()
        assert dim % n_head == 0
        self.d_k = dim // n_head
        self.num_heads = n_head
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        weights = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)

        p_attn = nn.functional.softmax(weights, dim=-1)

        if dropout is not None:
            p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, dim)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(dim, dropout) for _ in range(2)])

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, dim_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(nn.functional.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def subsequent_mask(batch, size):
    # mask out subsequent positions
    attn_shape = (batch, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0   