import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

from models.SeqPANLib.layers import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, Conv1D, SeqPANPredictor
from models.SeqPANLib.layers import DualAttentionBlock
from models.BaseLib.infer import infer_basic

class SeqPANBackBone(nn.Module):
    def __init__(self, configs, word_vectors):
        super().__init__()
        self.configs = configs
        dim = configs.model.dim
        droprate = configs.model.droprate
        max_pos_len = self. configs.model.max_vlen

        self.text_encoder = Embedding(num_words=configs.num_words, num_chars=configs.num_chars, out_dim=dim,
                                       word_dim=configs.model.word_dim, 
                                       char_dim=configs.model.char_dim, 
                                       word_vectors=word_vectors,
                                       droprate=droprate)

        self.tfeat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4, max_pos_len=max_pos_len, droprate=droprate)
        self.video_affine = VisualProjection(visual_dim=configs.model.vdim, dim=dim,droprate=droprate)
        self.vfeat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4,max_pos_len=max_pos_len, droprate=droprate)


        # self.dual_attention_block_1 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
        #                                                 droprate=droprate, use_bias=True, activation=None)
        # self.dual_attention_block_2 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
        #                                             droprate=droprate, use_bias=True, activation=None)

        self.q2v_attn = CQAttention(dim=dim, droprate=droprate)
        self.v2q_attn = CQAttention(dim=dim, droprate=droprate)
        self.cq_cat = CQConcatenate(dim=dim)
        self.predictor = SeqPANPredictor(configs)


    def forward(self, word_ids, char_ids, vfeat_in, vmask, tmask):
        torch.cuda.synchronize()
        start = time.time()
        
        
        B = vmask.shape[0]
        tfeat = self.text_encoder(word_ids, char_ids)
        vfeat = self.video_affine(vfeat_in)

        vfeat = self.vfeat_encoder(vfeat)
        tfeat = self.tfeat_encoder(tfeat)


        # vfeat_ = self.dual_attention_block_1(vfeat, tfeat, vmask, tmask)
        # tfeat_ = self.dual_attention_block_1(tfeat, vfeat, tmask, vmask)
        # vfeat, tfeat = vfeat_, tfeat_

        # vfeat_ = self.dual_attention_block_2(vfeat, tfeat, vmask, tmask)
        # tfeat_ = self.dual_attention_block_2(tfeat, vfeat, tmask, vmask)
        # vfeat, tfeat = vfeat_, tfeat_


        t2v_feat = self.q2v_attn(vfeat, tfeat, vmask, tmask)
        v2t_feat = self.v2q_attn(tfeat, vfeat, tmask, vmask)
        fuse_feat = self.cq_cat(t2v_feat, v2t_feat, tmask)
        slogits, elogits = self.predictor(fuse_feat, vmask)


        torch.cuda.synchronize()
        end = time.time()
        consume_time = end - start

        return {    "slogits": slogits,
                    "elogits": elogits,
                    "vmask" : vmask,
                    "consume_time": consume_time,
                    }

from utils.BaseDataset import BaseDataset, BaseCollate

class SeqPANBackBoneDataset(BaseDataset):
    def __init__(self, dataset, video_features, configs, loadertype):
        super().__init__(dataset, video_features, configs, loadertype)
    def __getitem__(self, index):
        res = BaseDataset.__getitem__(self, index)
        return res
    
class SeqPANBackBoneCollate(BaseCollate):
    def __call__(self, datas):
        return super().__call__(datas)
    


import time
def train_engine_SeqPANBackBone(model, data, configs, runmode):
    from models.BaseLib.loss import lossfun_loc, lossfun_match
    data = {key: value.to(configs.device) for key, value in data.items()}

    output = model(data['words_ids'], data['char_ids'], data['vfeats'], data['vmasks'], data['tmasks'])
    # location loss
    slogits = output["slogits"]
    elogits = output["elogits"]
    label_1Ds =  data['label_1Ds']
    loc_loss = lossfun_loc(slogits, elogits, label_1Ds[:, 0, :], label_1Ds[:, 1, :], data['vmasks'])


    loss =loc_loss 
    return loss, output


def infer_SeqPANBackBone(output, configs):

    start_logits = output["slogits"]
    end_logits = output["elogits"]
    vmask = output["vmask"]
    res = infer_basic(start_logits, end_logits, vmask)
    return res