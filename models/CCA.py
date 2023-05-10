import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import numpy as np
import pickle
from models.BaseLib.infer import infer_basic2d
from models.CCALib.layers import *

from models.SeqPANLib.layers import  WordEmbedding
from utils.BaseDataset import BaseDataset, BaseCollate
from models.BaseLib.layers import generate_2dmask

class CCA(nn.Module):
    def __init__(self, cfg, word_vectors):
        super(CCA, self).__init__()
        self.device = cfg.device

        self.word_emb = WordEmbedding(cfg.num_words, cfg.model.word_dim, 0, word_vectors=word_vectors)

        self.featpool = build_featpool(cfg) 
        self.feat2d = build_feat2d(cfg)
        self.simpredictor = build_simpredictor(cfg, self.feat2d.mask2d)
        self.T_fuse_attn = FuseAttention(cfg.MODEL.CCA.FEATPOOL.HIDDEN_SIZE, cfg.embed_size, True)
        self.C_GCN = C_GCN(cfg.num_attribute, in_channel=cfg.input_channel, t=0.3, embed_size=cfg.embed_size, adj_file=cfg.adj_file,
                                norm_func=cfg.norm_func_type, num_path=cfg.num_path, com_path=cfg.com_concept)
        
        self.v_t_param = nn.Parameter(torch.FloatTensor([0.5]))

        self.concept_dim = cfg.num_attribute
        # self.V_TransformerLayer = nn.TransformerEncoderLayer(cfg.MODEL.CCA.NUM_CLIPS, 8)
        self.V_TransformerLayer = nn.TransformerEncoderLayer(cfg.MODEL.CCA.NUM_CLIPS + self.concept_dim, 8)
        self.cut_dim = cfg.MODEL.CCA.NUM_CLIPS
    
    def forward(self, words_ids, tmask, vfeat_in, vmask, concept_input_embs):
        concept_basis = self.C_GCN(concept_input_embs)
        feats = self.featpool(vfeat_in)

        feats = torch.cat([feats, concept_basis.unsqueeze(0).repeat(feats.size(0), 1, 1).permute(0, 2, 1)], dim=2)
        feats = self.V_TransformerLayer(feats)[:, :, :self.cut_dim]
        map2d = self.feat2d(feats)

        tfeat = self.word_emb(words_ids)
        map2d_fused, queries = self.simpredictor(tfeat, tmask.sum(dim=1), map2d)

        queries_fused = self.T_fuse_attn(queries, concept_basis)
        # queries_fused = queries

        v2t_map2d = queries[:, :, None, None] * map2d_fused
        v2t_scores2d = torch.sum(F.normalize(v2t_map2d), dim=1).squeeze_()
        t2v_map2d = queries_fused[:, :, None, None] * map2d
        t2v_scores2d = torch.sum(F.normalize(t2v_map2d), dim=1).squeeze_()
        
        original_scores2d = self.v_t_param * v2t_scores2d + (1 - self.v_t_param) * t2v_scores2d
        
        res  = {"scores2d": original_scores2d,
                "vmask" : vmask}
        return res



class CCADataset(BaseDataset):
    def __init__(self, dataset, video_features, configs, loadertype):
        super().__init__(dataset, video_features, configs, loadertype)
        self.concept_input_embs = self.load_commonsense_emb(configs.attri_input_path, configs.commonsense_path)
        
    def __getitem__(self, index):
        res = BaseDataset.__getitem__(self, index)
        res["concept_input_embs"] = self.concept_input_embs
        return res
    
    def load_commonsense_emb(self, attri_input_path, commonsense_path):
        attribute_input_emb = pickle.load(open(attri_input_path, 'rb'))
        com_dict = pickle.load(open(commonsense_path, 'rb'))
        com_vectors = []
        for k in com_dict.keys():
            com_vectors.append(com_dict[k])
        com_vectors = np.array(com_vectors)
        attribute_input_emb = np.concatenate([attribute_input_emb, com_vectors], 0)
        attribute_input_emb = torch.from_numpy(attribute_input_emb)
        return attribute_input_emb 

class CCACollate(BaseCollate):
    def __call__(self, datas):
        res, records = super().__call__(datas)
        B = len(records)
        
        # concept_inputs = concept_input_embs[None, :, :].repeat(B, 1, 1)
        concept_inputs = []
        for d in datas:
            concept_inputs.append(d["concept_input_embs"])
        concept_inputs = torch.stack(concept_inputs)
        res["concept_inputs"] = concept_inputs
        return res, records
        # 'label_2Ds': label_2Ds,


def train_engine_CCA(model, data, configs, mode):
    from models.BaseLib.loss import lossfun_loc, lossfun_loc2d
    data = {key: value.to(configs.device) for key, value in data.items()}
    output = model(data['words_ids'], data['tmasks'], data['vfeats'], data['vmasks'], data['concept_inputs'])
    output["vmask"] = data['vmasks']
    # label_1Ds =  data['label_1Ds']
    label_2Ds =  data['label_2Ds']
    logit2D_mask = generate_2dmask(configs.MODEL.CCA.NUM_CLIPS).to(configs.device)

    lossfun_ccaloss = build_ccaloss(configs, logit2D_mask)
    loc_2dloss = lossfun_ccaloss(output["scores2d"], label_2Ds)
    
    loss = loc_2dloss 
    return loss, output

def infer_CCA(output, configs):
    scores2d, vmask = output["scores2d"], output['vmask']
    logit2D_mask = generate_2dmask(configs.MODEL.CCA.NUM_CLIPS).to(configs.device)
    res = infer_basic2d(scores2d, logit2D_mask, vmask)
    return res
