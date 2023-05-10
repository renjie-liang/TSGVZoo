import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np


from models.SeqPANLib.layers import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, Conv1D, SeqPANPredictor
from models.SeqPANLib.layers import DualAttentionBlock, WordEmbedding
from models.CPLLib.layers import TransformerDecoder, _generate_mask, SinusoidalPositionalEmbedding

class CPL(nn.Module):
    def __init__(self, configs, word_vectors):
        super(CPL, self).__init__()
        self.configs = configs
        dim = configs.model.dim
        self.num_props = configs.model.num_props
        droprate = configs.model.droprate
        self.vocab_size = configs.num_words

        max_pos_len = self. configs.model.max_vlen
        self.text_encoder = Embedding(num_words=configs.num_words, num_chars=configs.num_chars, out_dim=dim,
                                       word_dim=configs.model.word_dim, 
                                       char_dim=configs.model.char_dim, 
                                       word_vectors=word_vectors,
                                       droprate=droprate)
                                       
        self.video_affine = VisualProjection(visual_dim=configs.model.vdim, dim=dim,
                                             droprate=droprate)
        self.feat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4,
                                              max_pos_len=max_pos_len, droprate=droprate)


        self.dual_attention_block_1 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
                                                        droprate=droprate, use_bias=True, activation=None)
        self.dual_attention_block_2 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
                                                    droprate=droprate, use_bias=True, activation=None)



        # self.tfeat_encoder = FeatureEncoder(dim=dim, num_heads=configs.model.num_heads, kernel_size=7, num_layers=4,
        #                                       max_pos_len=max_pos_len, droprate=droprate)

        self.q2v_attn = CQAttention(dim=dim, droprate=droprate)
        self.v2q_attn = CQAttention(dim=dim, droprate=droprate)
        self.cq_cat = CQConcatenate(dim=dim)
        self.match_conv1d = Conv1D(in_dim=dim, out_dim=4)

        lable_emb = torch.empty(size=[dim, 4], dtype=torch.float32)
        lable_emb = torch.nn.init.orthogonal_(lable_emb.data)
        self.label_embs = nn.Parameter(lable_emb, requires_grad=True)
        
        self.predictor = SeqPANPredictor(configs)

        self.decoder1 = TransformerDecoder(num_layers=2, d_model=dim, num_heads=4, dropout=0.1)
        self.decoder2 = TransformerDecoder(num_layers=2, d_model=dim, num_heads=4, dropout=0.1)
        self.word_emb = WordEmbedding(configs.num_words, configs.model.word_dim, 0.0, word_vectors=word_vectors)
        self.word_fc = nn.Linear(configs.model.word_dim, dim)
        self.start_vec = nn.Parameter(torch.zeros(configs.model.word_dim).float(), requires_grad=True)
        self.word_pos_encoder = SinusoidalPositionalEmbedding(dim, 0, 20)
        self.conv1d_cw = nn.Conv1d(in_channels=max_pos_len, out_channels=1, kernel_size=1)

        self.fc_gauss = nn.Linear(dim, self.num_props*2)
        self.fc_comp = nn.Linear(dim, self.vocab_size)

    def forward(self, word_ids, vfeat_in, vmask, tmask):
        B, L, D = vfeat_in.shape
        P = self.num_props
        # tfeat = self.text_encoder(word_ids, char_ids)
        vfeat_tmp = self.video_affine(vfeat_in)

        #### CPL
        words_feat = self.word_emb(word_ids)
        tmp = torch.zeros([B, 1, words_feat.shape[-1]]).cuda()
        words_feat = torch.concat([tmp, words_feat], dim=1)
        words_feat[:, 0] = self.start_vec
        words_feat = F.dropout(words_feat, 0.1, self.training)
        tfeat_long = self.word_fc(words_feat)
        tmask_long = _generate_mask(words_feat, tmask.sum(dim=1).long() + 1)

        props_len = L

        weakly_feat = self.conv1d_cw(vfeat_tmp).squeeze(1)
        gauss_param = torch.sigmoid(self.fc_gauss(weakly_feat)).view(B*P, 2)
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1]
        
        vfeat_props = torch.repeat_interleave(vfeat_tmp, P, dim=0)
        vmask_props = torch.repeat_interleave(vmask, P, dim=0)
        gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width, vmask_props)
        pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]

        tmask_props = torch.repeat_interleave(tmask_long[:, :-1], P, dim=0)
        tfeat_props = torch.repeat_interleave(tfeat_long[:, :-1], P, dim=0)

        enc_out, _ = self.decoder1(None, None, vfeat_props, vmask_props, tgt_gauss_weight=pos_weight)
        out, weight = self.decoder2(enc_out, vmask_props, tfeat_props, tmask_props, src_gauss_weight=pos_weight)
        words_logit = self.fc_comp(out)

        res = {
            'word_ids': word_ids,
            'words_mask': tmask_long[:, :-1],
            'words_logit': words_logit,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
        }


        return res



    def generate_gauss_weight(self, props_len, center, width, vmask):
        # pdb.set_trace()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        center = center * ( vmask.sum(dim=1) / vmask.shape[1])
        center = center.unsqueeze(-1)
        width = width * ( vmask.sum(dim=1) / vmask.shape[1])
        width = width.unsqueeze(-1).clamp(1e-2) / 9

        w = 0.3989422804014327
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]





# def train_engine_CPL(model, data, configs):
#     device = configs.device
#     _, vfeats, vmask, word_ids, char_ids, tmask, s_labels, e_labels, m_labels = data
#     # prepare features
#     vfeats, vmask = vfeats.to(device), vmask.to(device) ### move_to_cuda
#     word_ids, char_ids, tmask = word_ids.to(device), char_ids.to(device), tmask.to(device)
#     s_labels, e_labels, m_labels = s_labels.to(device), e_labels.to(device), m_labels.to(device)

#     # # compute logits
#     output= model(word_ids, char_ids, vfeats, vmask, tmask)


#     loss_rec = rec_loss_cpl(configs=configs, tlogist_prop=output["words_logit"], 
#                         words_id= output["word_ids"], words_mask=output["words_mask"], tlogist_gt=None)
#     loss_div = div_loss_cpl(words_logit=output["words_logit"], gauss_weight= output["gauss_weight"], configs=configs)
#     loss = loss_rec + loss_div

#     output["vmask"] = vmask
#     return loss, output




# def infer_CPL(output, configs): ## don't consider vmask
#     from models.cpl_lib import cal_nll_loss

#     P = configs.others.cpl_num_props
#     B = output['words_logit'].size(0)  // P

#     tmask_props = torch.repeat_interleave(output['words_mask'], P, dim=0)
#     word_ids = torch.repeat_interleave(output['word_ids'], P, dim=0)
#     nll_loss, acc = cal_nll_loss(output['words_logit'], word_ids, tmask_props)
#     idx = nll_loss.view(B, P).argsort(dim=-1)

#     width = output['width'].view(B, P).gather(index=idx, dim=-1)
#     center = output['center'].view(B, P).gather(index=idx, dim=-1)
    
#     selected_props = torch.stack([torch.clamp(center-width/2, min=0), 
#                                     torch.clamp(center+width/2, max=1)], dim=-1)
#     selected_props = selected_props.detach().cpu().numpy()
#     start_fracs, end_fracs = selected_props[:, 0, 0], selected_props[:, 0, 1] 

#     return start_fracs, end_fracs
        

