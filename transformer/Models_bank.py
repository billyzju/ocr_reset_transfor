''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism.
        n_src_vocab: 表示输入数据集词库的大小
        len_max_seq: 表示输入数据集的最大句子长度
        d_word_vec: 表示embedding后的词向量的维度
        n_layer: 表示transformer有几层
        n_head: 表示有几个head
        d_k: KEY的维度
        d_v: VALUE的维度
        d_model: 模型内部的维度
        d_inner: ff第二层的维度

    '''

    ## def __init__(
    ##         self,
    ##         n_src_vocab, len_max_seq, d_word_vec,
    ##         n_layers, n_head, d_k, d_v,
    ##         d_model, d_inner, dropout=0.1):
    def __init__(self, encoder_input_len, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        '''
        encoder_input_len: 卷积输出的第二维度的大小
        n_layers:
        n_head:
        d_k:
        d_v:
        d_model:
        d_inner:
        dropout:
        '''
        super().__init__()

        n_position = encoder_input_len + 1

        ##self.src_word_emb = nn.Embedding(
        ##    n_src_vocab, d_word_vec, padding_idx=Constants.PAD) # 建立encoder端的词向量矩阵 [n_src_vocab-词库大小，d_word_vec-词向量维度]

        ##self.position_enc = nn.Embedding.from_pretrained(
        ##   get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
        ##   freeze=True) #建立position_embedding矩阵 [n_position-最大句子长度，d_word_vec-词向量维度]

        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(n_position, d_model), freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)]) #建立encoder层，d_model为结构中通用的维度，d_k为key的最终维度，d_v为value的最终维度

    ##def forward(self, src_seq, src_pos, return_attns=False):
    def forward(self, encoder_input, encoder_input_len, return_attns=False):
        '''src_seq 输入的句子
           src_pos 应该是一个index_list类似于[0, 1, 2, 3]
        '''

        enc_slf_attn_list = []

        # -- Prepare masks
        ##slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq) #获得attention_mask，形状为[batch_size, q_len, k_len]
        ##non_pad_mask = get_non_pad_mask(src_seq) #获得pad_mask, 形状为[batch_size, q_len, 1]

        # -- Forward
        ##enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos) #生成word_vector并且加上position_info
        batch_size = encoder_input.size()[0]
        slf_attn_mask = 1 - torch.ones(batch_size, encoder_input_len, encoder_input_len)
        non_pad_mask = torch.ones(batch_size, encoder_input_len, 1)
		
        slf_attn_mask = slf_attn_mask.cuda()
        non_pad_mask = non_pad_mask.cuda()

        pos = torch.arange(1, encoder_input_len+1).unsqueeze(0).expand(batch_size, -1).cuda()
        enc_output = encoder_input + self.position_enc(pos)
        #enc_output = encoder_input

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism.
        n_tgt_vocab: decoder输入的词库大小，
        len_max_seq: 句子的最大长度，
        d_word_vec:  word_vector后的维度，

    '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    ##def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
    def forward(self, tgt_seq, enc_output, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq).cuda()

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq) #获得上三角为1，下三角包括中轴线为0的矩阵，形状为[batch_size, len, len]
        slf_attn_mask_keypad =  get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0).cuda()

        ##dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)
        batch_size = enc_output.size()[0]
        len_k = enc_output.size()[1]
        len_q = tgt_seq.size()[1]
        dec_enc_attn_mask = (1 - torch.ones(batch_size, len_q, len_k)).cuda()
        # -- Forward
        tgt_pos_a = torch.arange(1, len_q+1).unsqueeze(0).expand(batch_size, -1).cuda()
        tgt_pos_b = tgt_seq.eq(Constants.PAD).cuda()
        tgt_pos = tgt_pos_a.masked_fill(tgt_pos_b, Constants.PAD)

        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism.
    n_src_vocab: encoder端输入句子的词库的大小，
    n_tgt_vocab: decoder端输入句子的词库的大小，
    len_max_seq: 最大句子长度，
    d_word_vec: 生成的词向量的维度，
    d_model: transform模型内部的维度，
    d_innder: ff中间层的矢量维度，


    '''

    ## def __init__(
    ##         self,
    ##         n_src_vocab, n_tgt_vocab, len_max_seq,
    ##         d_word_vec=512, d_model=512, d_inner=2048,
    ##         n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
    ##         tgt_emb_prj_weight_sharing=True,
    ##         emb_src_tgt_weight_sharing=True):
    def __init__(self, encoder_input_len, n_tgt_vocab, len_max_seq,
                 d_word_vec=512, d_model=512, d_inner=2048,
                 n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
                 tgt_emb_prj_weight_sharing=True,
                 emb_src_tgt_weight_sharing=False):

        super().__init__()

        ## self.encoder = Encoder(
        ##     n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
        ##     d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
        ##     n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
        ##     dropout=dropout)
        self.encoder = Encoder(encoder_input_len=encoder_input_len, d_model=d_model, d_inner=d_inner,
                               n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    ##def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
    def forward(self, encoder_input, encoder_input_len, tgt_seq):
        ##tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        ##enc_output, *_ = self.encoder(src_seq, src_pos)
        ##dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        enc_output, *_ = self.encoder(encoder_input, encoder_input_len)
        dec_output, *_ = self.decoder(tgt_seq, enc_output)

        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
