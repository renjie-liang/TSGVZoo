import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

from models.SeqPANLib.layers import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, Conv1D, SeqPANPredictor
from models.SeqPANLib.layers import DualAttentionBlock
from utils.BaseDataset import BaseDataset, BaseCollate
from models.BertLib.layers import BertEmbedding, BertEmbedding2, RoBERTaEmbedding
from transformers import BertTokenizer, RobertaTokenizer
from models.BaseLib.infer import infer_basic

class SeqPANBert(nn.Module):
    def __init__(self, configs, word_vectors):
        super(SeqPANBert, self).__init__()
        self.configs = configs
        dim = configs.model.dim
        droprate = configs.model.droprate
        max_pos_len = self.configs.model.max_vlen
        
        
        if configs.model.text_embeding == "RoBERTa":
            self.bert_encoder = RoBERTaEmbedding(indim=768, outdim=dim, droprate=droprate)
        elif configs.model.text_embeding == "BERT":
            self.bert_encoder = BertEmbedding(indim=768, outdim=dim, droprate=droprate)
        else: 
            raise
        
        self.tfeat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4, max_pos_len=max_pos_len, droprate=droprate)
                                       
        self.video_affine = VisualProjection(visual_dim=configs.model.vdim, dim=dim,
                                             droprate=droprate)
        self.vfeat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4,
                                              max_pos_len=max_pos_len, droprate=droprate)


        self.dual_attention_block_1 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
                                                        droprate=droprate, use_bias=True, activation=None)
        self.dual_attention_block_2 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
                                                    droprate=droprate, use_bias=True, activation=None)


        self.q2v_attn = CQAttention(dim=dim, droprate=droprate)
        self.v2q_attn = CQAttention(dim=dim, droprate=droprate)
        self.cq_cat = CQConcatenate(dim=dim)
        self.match_conv1d = Conv1D(in_dim=dim, out_dim=4)

        lable_emb = torch.empty(size=[dim, 4], dtype=torch.float32)
        lable_emb = torch.nn.init.orthogonal_(lable_emb.data)
        self.label_embs = nn.Parameter(lable_emb, requires_grad=True)
        self.predictor = SeqPANPredictor(configs)


    def forward(self, bert_ids, vfeat_in, vmask, tmask):
        B = vmask.shape[0]
        tfeat= self.bert_encoder(bert_ids, tmask)
        tfeat = self.tfeat_encoder(tfeat)
        
        vfeat = self.video_affine(vfeat_in)
        vfeat = self.vfeat_encoder(vfeat)

        vfeat_ = self.dual_attention_block_1(vfeat, tfeat, vmask, tmask)
        tfeat_ = self.dual_attention_block_1(tfeat, vfeat, tmask, vmask)
        vfeat, tfeat = vfeat_, tfeat_

        vfeat_ = self.dual_attention_block_2(vfeat, tfeat, vmask, tmask)
        tfeat_ = self.dual_attention_block_2(tfeat, vfeat, tmask, vmask)
        vfeat, tfeat = vfeat_, tfeat_

        t2v_feat = self.q2v_attn(vfeat, tfeat, vmask, tmask)
        v2t_feat = self.v2q_attn(tfeat, vfeat, tmask, vmask)
        fuse_feat = self.cq_cat(t2v_feat, v2t_feat, tmask)

        match_logits = self.match_conv1d(fuse_feat)
        match_score = F.gumbel_softmax(match_logits, tau=0.3)
        match_probs =torch.log(match_score)
        soft_label_embs = torch.matmul(match_score, torch.tile(self.label_embs, (B, 1, 1)).permute(0, 2, 1))
        fuse_feat = (fuse_feat + soft_label_embs) * vmask.unsqueeze(2)
        slogits, elogits = self.predictor(fuse_feat, vmask)

        return {    "slogits": slogits,
                    "elogits": elogits,
                    "vmask" : vmask,
                    "match_score" : match_score,
                    "label_embs" : self.label_embs,
                    }


class SeqPANBertDataset(BaseDataset):
    def __init__(self, dataset, video_features, configs, loadertype):
        super().__init__(dataset, video_features, configs, loadertype)
        self.max_tlen = configs.model.max_tlen
        if configs.model.text_embeding == "RoBERTa":
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif configs.model.text_embeding == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else: 
            raise
        
        
        
    def __getitem__(self, index):
        res = BaseDataset.__getitem__(self, index)
        sentence = res['record']['sentence']
        bert_input = self.tokenizer(sentence, padding='max_length', max_length = self.max_tlen,  truncation=True, return_tensors="pt")
        bert_id = bert_input["input_ids"]
        bert_mask = bert_input["attention_mask"]

        res['bert_id'] = bert_id
        res['bert_mask'] = bert_mask
        return res
    
class SeqPANBertCollate(BaseCollate):
    def __call__(self, datas):
        res, records = super().__call__(datas)
        bert_ids, bert_masks = [], []
        for r in datas:
            bert_ids.append(r["bert_id"])
            bert_masks.append(r["bert_mask"])
        res['bert_ids'] = torch.vstack(bert_ids)
        res['bert_masks'] = torch.vstack(bert_masks)  
        return res, records


import time
def train_engine_SeqPANBert(model, data, configs, runtype):
    from models.BaseLib.loss import lossfun_loc, lossfun_match
    data = {key: value.to(configs.device) for key, value in data.items()}
    output = model(data['bert_ids'], data['vfeats'], data['vmasks'], data['bert_masks'])
    slogits = output["slogits"]
    elogits = output["elogits"]
    label_1Ds =  data['label_1Ds']
    loc_loss = lossfun_loc(slogits, elogits, label_1Ds[:, 0, :], label_1Ds[:, 1, :], data['vmasks'])
    m_loss = lossfun_match(output["match_score"], output["label_embs"],  data["NER_labels"],  data['vmasks'])

    loss =loc_loss + m_loss
    return loss, output


def infer_SeqPANBert(output, configs):
    start_logits = output["slogits"]
    end_logits = output["elogits"]
    vmask = output["vmask"]
    res = infer_basic(start_logits, end_logits, vmask)
    return res