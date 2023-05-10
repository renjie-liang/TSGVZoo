
import torch.nn as nn
from transformers import BertModel
from models.BaseLib.layers import LinearBlock
from models.SeqPANLib.layers import Conv1D


class BertEmbedding(nn.Module):
    def __init__(self, indim, outdim, droprate):
        super().__init__()
        
        self.bert_emb = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_emb.parameters():
            param.requires_grad = False
        # self.fc = Conv1D(in_dim=indim, out_dim=outdim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc = LinearBlock(indim=indim, outdim=outdim, droprate=droprate)
        
    def forward(self, bert_ids, tmask):
        emb, _ = self.bert_emb(input_ids=bert_ids, attention_mask=tmask, return_dict=False)
        emb = self.fc(emb) 
        return emb



class BertEmbedding2(nn.Module):
    def __init__(self, indim, outdim, droprate):
        super().__init__()
        self.bert_emb = BertModel.from_pretrained('bert-base-uncased')
        self.query_conv1d = Conv1D(in_dim=indim, out_dim=outdim, kernel_size=1, stride=1, padding=0, bias=True)
        self.q_layer_norm = nn.LayerNorm(outdim, eps=1e-6)
    def forward(self, bert_ids, tmask):
        emb, _ = self.bert_emb(input_ids=bert_ids, attention_mask=tmask, return_dict=False)
        emb = self.query_conv1d(emb) 
        emb = self.q_layer_norm(emb)
        return emb
    

from transformers import RobertaModel
class RoBERTaEmbedding(nn.Module):
    def __init__(self, indim, outdim, droprate):
        super().__init__()
        self.bert_emb = RobertaModel.from_pretrained('roberta-base')
        for param in self.bert_emb.parameters():
            param.requires_grad = False
        
        self.query_conv1d = Conv1D(in_dim=indim, out_dim=outdim, kernel_size=1, stride=1, padding=0, bias=True)
        self.q_layer_norm = nn.LayerNorm(outdim, eps=1e-6)
    def forward(self, bert_ids, tmask):
        emb, _ = self.bert_emb(input_ids=bert_ids, attention_mask=tmask, return_dict=False)
        emb = self.query_conv1d(emb) 
        emb = self.q_layer_norm(emb)
        return emb