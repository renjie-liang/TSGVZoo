import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.BaseLib.layers import mask_logits

def infer_basic(start_logits, end_logits, vmask):
    L = start_logits.shape[1]
    start_logits = mask_logits(start_logits, vmask)
    end_logits = mask_logits(end_logits, vmask)

    start_prob = torch.softmax(start_logits, dim=1) ### !!!
    end_prob = torch.softmax(end_logits, dim=1)
    
    outer = torch.matmul(start_prob.unsqueeze(2),end_prob.unsqueeze(1))
    outer = torch.triu(outer, diagonal=0)
    _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
    _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
    
    sfrac = (start_index/vmask.sum(dim=1)).cpu().numpy()
    efrac = (end_index/vmask.sum(dim=1)).cpu().numpy()
    res = np.stack([sfrac, efrac]).T
    return res

def infer_basic2d(scores2d, logit2D_mask, vmask):
    scores2d = scores2d.sigmoid_() * logit2D_mask

    outer = torch.triu(scores2d, diagonal=0)
    _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
    _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
    
    sfrac = (start_index/vmask.sum(dim=1)).cpu().numpy()
    efrac = (end_index/vmask.sum(dim=1)).cpu().numpy()
    res = np.stack([sfrac, efrac]).T
    return res
