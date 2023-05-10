import torch
import torch.nn as nn
# from models.layers import Embedding, VisualProjection, CQAttention, CQConcatenate
from models.VSLNetLib.layers_vsl import *
from utils.BaseDataset import BaseDataset, BaseCollate
from models.BaseLib.infer import infer_basic

class VSLNet(nn.Module):
    def __init__(self, configs, word_vectors):
        super(VSLNet, self).__init__()
        self.configs = configs
        dim = configs.model.dim
        droprate = configs.model.droprate
        self.embedding_net = Embedding(num_words=configs.num_words, num_chars=configs.num_chars, out_dim=dim,
                                       word_dim=configs.model.word_dim, char_dim=configs.model.char_dim, word_vectors=word_vectors,
                                       drop_rate=droprate)
        self.video_affine = VisualProjection(visual_dim=configs.model.vdim, dim=dim,
                                             drop_rate=droprate)
        self.feature_encoder = FeatureEncoder(dim=dim, num_heads=configs.model.num_heads, kernel_size=7, num_layers=4,
                                              max_pos_len=configs.model.max_vlen, drop_rate=droprate)
        # video and query fusion
        self.cq_attention = CQAttention(dim=dim, drop_rate=droprate)
        self.cq_concat = CQConcatenate(dim=dim)
        # query-guided highlighting
        self.highlight_layer = HighLightLayer(dim=dim)
        # conditioned predictor
        self.predictor = ConditionedPredictor(dim=dim, num_heads=configs.model.num_heads, drop_rate=droprate,
                                              max_pos_len=configs.model.max_vlen, predictor=configs.model.predictor)
        # init parameters
        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()
        self.apply(init_weights)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        video_features = self.video_affine(video_features)
        query_features = self.embedding_net(word_ids, char_ids)
        video_features = self.feature_encoder(video_features, mask=v_mask)
        query_features = self.feature_encoder(query_features, mask=q_mask)
        features = self.cq_attention(video_features, query_features, v_mask, q_mask)
        features = self.cq_concat(features, query_features, q_mask)
        h_score = self.highlight_layer(features, v_mask)
        features = features * h_score.unsqueeze(2)
        slogits, elogits = self.predictor(features, mask=v_mask)
        return {    "slogits": slogits,
                    "elogits": elogits,
                    "vmask" : v_mask,
                    "h_score" : h_score,
                    }







class VSLNetDataset(BaseDataset):
    def __init__(self, dataset, video_features, configs, loadertype):
        super().__init__(dataset, video_features, configs, loadertype)
    def __getitem__(self, index):
        res = BaseDataset.__getitem__(self, index)
        return res
    
class VSLNetCollate(BaseCollate):
    def __call__(self, datas):
        return super().__call__(datas)



def train_engine_VSLNet(model, data, configs, runtype):
    from models.BaseLib.loss import lossfun_loc, loss_highlight
    data = {key: value.to(configs.device) for key, value in data.items()}
    output = model(data['words_ids'], data['char_ids'], data['vfeats'], data['vmasks'], data['tmasks'])
    slogits, elogits, h_score = output["slogits"], output["elogits"], output["h_score"]
    label_1Ds =  data['label_1Ds']
    loc_loss = lossfun_loc(slogits, elogits, label_1Ds[:, 0, :], label_1Ds[:, 1, :], data['vmasks'])

    hlabel = data['NER_labels']
    hlabel[hlabel==2] = 1
    hlabel[hlabel==3] = 1
    highlight_loss = loss_highlight(h_score, hlabel, data["vmasks"])
    loss =loc_loss +  highlight_loss * configs.loss.highlight_lambda
    return loss, output


def infer_VSLNet(output, configs):
    start_logits = output["slogits"]
    end_logits = output["elogits"]
    vmask = output["vmask"]
    res = infer_basic(start_logits, end_logits, vmask)
    return res