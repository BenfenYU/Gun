import torch, math, numpy
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from config import Config
from torch.nn import init
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


config = Config()

class SmoothedCrossEntropyLoss(torch.nn.Module):

    def __init__(self, smoothing=0.3, normalize=True):
        super(SmoothedCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.normalize = normalize
        
    def forward(self, logits, labels):
        shape = labels.shape
        logits = torch.reshape(logits, [-1, logits.shape[-1]])
        labels = torch.reshape(labels, [-1])

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        batch_idx = torch.arange(labels.shape[0], device=logits.device)
        loss = log_probs[batch_idx, labels]

        if not self.smoothing:
            return -torch.reshape(loss, shape)

        n = logits.shape[-1] - 1.0
        p = 1.0 - self.smoothing
        q = self.smoothing / n

        if log_probs.dtype != torch.float16:
            sum_probs = torch.sum(log_probs, dim=-1)
            loss = p * loss + q * (sum_probs - loss)
        else:
            sum_probs = torch.sum(log_probs.to(torch.float32), dim=-1)
            loss = loss.to(torch.float32)
            loss = p * loss + q * (sum_probs - loss)
            loss = loss.to(torch.float16)

        loss = -torch.reshape(loss, shape)

        if self.normalize:
            normalizing = -(p * math.log(p) + n * q * math.log(q + 1e-20))
            return torch.sum(loss - normalizing)
        else:
            return loss


class birnn(torch.nn.Module):
    def __init__(self, rnn_type):
        super(birnn, self).__init__()
        self.word_emb_dim = config.word_emb_dim
        self.pos_emb_dim = config.pos_emb_dim
        self.n_voc = config.n_voc
        self.n_tag = config.n_tag
        self.n_pos = config.n_pos
        self.hidden_dim = config.hidden_dim
        self.rnn_type = rnn_type
        self.input_emb_dim = self.word_emb_dim + self.pos_emb_dim
        self.mname = "birnn"


        #self.cri = torch.nn.CrossEntropyLoss(weight=config.weight)  # 返回标量
        self.cri = SmoothedCrossEntropyLoss()


        # embedding
        self.word_embbedding = torch.nn.Embedding(self.n_voc, self.word_emb_dim)
        self.pos_embbedding = torch.nn.Embedding(self.n_pos, self.pos_emb_dim)

        #biRnn
        if self.rnn_type == 'l':
            self.binet = torch.nn.LSTM(input_size=self.input_emb_dim,
                                       hidden_size=self.hidden_dim,
                                       num_layers=8, bidirectional=True,
                                       batch_first=True)
            
        elif self.rnn_type == 'g':
            self.binet = torch.nn.GRU(input_size=self.input_emb_dim,
                                      hidden_size=self.hidden_dim,
                                      num_layers=1, bidirectional=True,
                                      batch_first=True)
        else:
            print('网络类型不符合要求')

        self.lrelu = torch.nn.LeakyReLU(negative_slope = 0.01)
        self.lin = torch.nn.Linear(self.hidden_dim*2, self.n_tag)
        #self.output = torch.nn.Softmax(dim = -1)

    # 拼接词向量
    def cat_emb(self, word_embs, pos_embs):
        # [batch_size, seq_len, emb_dim]
        return torch.cat((word_embs, pos_embs), dim=2)

    def forward(self, batch, pos, seq_len,pack_pad = True):

        '''正向传播，输出每句话预测标记,words,property,role'''
        word_embs = self.word_embbedding(batch)
        pos_embs = self.pos_embbedding(pos)
        embs = self.cat_emb(word_embs, pos_embs)

        if(pack_pad):

            pack_input = pack_padded_sequence(input=embs, lengths=seq_len,
                                            batch_first=True, enforce_sorted=False)
            pack_output, _ = self.binet(pack_input)
            pad_packed_output, _ = pad_packed_sequence(pack_output, batch_first=True)
            scores = self.lin(pad_packed_output)
        else:
            pack_output, _ = self.binet(embs)
            scores = self.lin(pack_output)

        return scores  # [batch_size, seq_len, n_tag]

    def decode(self, batch, pos, seq_len):

        # word_embs = self.word_embbedding(batch).to(config.device)
        # pos_embs = self.pos_embbedding(pos).to(config.device)
        # embs = self.cat_emb(word_embs, pos_embs).to(config.device)
        #
        # pack_input = pack_padded_sequence(input=embs, lengths=seq_len,
        #                                   batch_first=True, enforce_sorted=False)
        # pack_output, _ = self.binet(pack_input)
        # pad_packed_output, _ = pad_packed_sequence(pack_output)

        # scores = self.output(pad_packed_output).to(config.device)
        scores = self.forward(batch, pos, seq_len)
        #scores = torch.softmax(scores, dim=2)
        all_res = []
        for batch in scores:
            sub_res = []

            for words in batch:
                words = words.cpu().detach().numpy().tolist()
                max_pro = max(words)
                index_pro = words.index(max_pro)
                sub_res.append(index_pro)

            all_res.append(sub_res)

        return all_res

    def get_loss(self, input, tags):
        #loss_list = []
        batch_size, seq_len, n_tag = input.size()

        loss_ = torch.zeros(1, dtype=torch.float).to(config.device)
        for item in range(batch_size):
            c_input = input[item, :, :]
            c_tags = tags[item, :]
            loss = self.cri(c_input,c_tags)
            loss_ += loss
            #loss_list.append(loss.unsqueeze(0))  # 不升维，o维不能拼接
        #loss_batch = torch.cat([item for item in loss_list]).sum()

        # input = input.reshape(input.shape[0]*input.shape[1], -1)
        # tags = tags.reshape(-1)
        #
        # loss_ = self.cri(input, tags)
        return loss_

