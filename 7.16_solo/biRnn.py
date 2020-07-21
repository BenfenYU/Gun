import torch, math, numpy
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from config import Config
from torch.nn import init

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

    def forward(self, batch, pos, seq_len):

        '''正向传播，输出每句话预测标记,words,property,role'''
        word_embs = self.word_embbedding(batch)
        pos_embs = self.pos_embbedding(pos)
        embs = self.cat_emb(word_embs, pos_embs)

        #pack_input = pack_padded_sequence(input=embs, lengths=seq_len,
                                        #batch_first=True, enforce_sorted=False)
        pack_output, _ = self.binet(embs)
        #pad_packed_output, _ = pad_packed_sequence(pack_output, batch_first=True)

        #lrelu = self.lrelu(pack_output)
        #lin = self.lin(lrelu)
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


class BiLSTM_CRF(torch.nn.Module):
    def __init__(self, vocab_size = config.n_voc, hidden_size = config.hidden_dim, out_size = config.n_tag):
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = birnn()

        # CRF实际上就是多学习一个转移矩阵 [out_size, out_size] 初始化为均匀分布
        self.transition = nn.Parameter(
            torch.ones(out_size, out_size) * 1/out_size)
        # self.transition.data.zero_()
        self.cri = SmoothedCrossEntropyLoss()

    def forward(self, sents_tensor, lengths):
        # [B, L, out_size]
        emission = self.bilstm(sents_tensor, lengths)
        batch_size, max_len, out_size = emission.size()
        crf_scores = emission.unsqueeze(
            2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)

        return crf_scores
    
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

    def test(self, test_sents_tensor, lengths, tag2id):
        """使用维特比算法进行解码"""
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        crf_scores = self.forward(test_sents_tensor, lengths)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                        :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1]（L-1是因为扣去了end_token),大小的liebiao
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids
