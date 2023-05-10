import torch.nn.functional as F
import torch
from torch import nn
import numpy as np


# def lossfun_match(m_probs, label_embs, m_labels, vmask):
    
#     ## cross_entropy????
#     m_labels = F.one_hot(m_labels)
#     loss_per_sample = -torch.sum(m_labels * m_probs, dim=-1)
#     m_loss =torch.sum(loss_per_sample * vmask, dim=-1) / (torch.sum(vmask, dim=-1) + 1e-12)
#     m_loss = m_loss.mean()
    
#     # add punishment
#     ortho_constraint = torch.matmul(label_embs.T, label_embs) * (1.0 - torch.eye(4, device=label_embs.device, dtype=torch.float32))
#     ortho_constraint = torch.norm(ortho_constraint, p=2)  # compute l2 norm as loss
#     m_loss += ortho_constraint

#     return m_loss

def lossfun_match(m_probs, label_embs, m_labels, vmask):
    # NLLLoss
    # loss_fun = nn.NLLLoss()
    # loss_fun = nn.CrossEntropyLoss()
    m_labels = F.one_hot(m_labels).float()
    # m_loss = loss_fun(m_probs, m_labels)
    # m_loss = loss_fun(m_probs.transpose(1,2), m_labels)

    loss_per_sample = -torch.sum(m_labels * m_probs, dim=-1)
    # m_loss =torch.sum(loss_per_sample * vmask, dim=-1) / (torch.sum(vmask, dim=-1) + 1e-12)
    # m_loss = m_loss.mean()
    m_loss =torch.sum(loss_per_sample * vmask) / (torch.sum(vmask) + 1e-12)
    
    # add punishment
    ortho_constraint = torch.matmul(label_embs.T, label_embs) * (1.0 - torch.eye(4, device=label_embs.device, dtype=torch.float32))
    ortho_constraint = torch.norm(ortho_constraint, p=2)  # compute l2 norm as loss
    m_loss += ortho_constraint
    return m_loss

def lossfun_loc(start_logits, end_logits, s_labels, e_labels, vmask):
    start_logits = start_logits * vmask
    end_logits = end_logits * vmask

    sloss = nn.CrossEntropyLoss(reduction='mean')(start_logits, s_labels)
    eloss = nn.CrossEntropyLoss(reduction='mean')(end_logits, e_labels)
    loss = sloss + eloss
    return loss

# VSLNet
def loss_highlight(scores, labels, mask, epsilon=1e-12):
    labels = labels.type(torch.float32)
    weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
    loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
    loss_per_location = loss_per_location * weights
    mask = mask.type(torch.float32)
    loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
    return loss




# ### CPL
def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()
    
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc


def rec_loss_cpl(configs, tlogist_prop, words_id, words_mask, tlogist_gt=None):
    P = configs.others.cpl_num_props
    B = tlogist_prop.size(0) // P

    words_mask1 = words_mask.unsqueeze(1) \
        .expand(B, P, -1).contiguous().view(B*P, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(B, P, -1).contiguous().view(B*P, -1)

    nll_loss, acc = cal_nll_loss(tlogist_prop, words_id1, words_mask1)
    nll_loss = nll_loss.view(B, P)
    min_nll_loss = nll_loss.min(dim=-1)[0]

    final_loss = min_nll_loss.mean()

    # if not tlogist_gt:
    #     ref_nll_loss, ref_acc = cal_nll_loss(tlogist_gt, words_id, words_mask) 
    #     final_loss = final_loss + ref_nll_loss.mean()
    #     final_loss = final_loss / 2

    return final_loss


def div_loss_cpl(words_logit, gauss_weight, configs):
    P = configs.others.cpl_num_props
    B = words_logit.size(0) // P
    
    gauss_weight = gauss_weight.view(B, P, -1)
    gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
    target = torch.eye(P).unsqueeze(0).cuda() * configs.others.cpl_div_lambda
    source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
    div_loss = torch.norm(target - source, dim=(1, 2))**2

    return div_loss.mean() * configs.others.cpl_div_loss_alhpa


def lossfun_loc2d(scores2d, labels2d, mask2d):
    def scale(iou, min_iou, max_iou):
        return (iou - min_iou) / (max_iou - min_iou)

    labels2d = scale(labels2d, 0.5, 1.0).clamp(0, 1)
    loss_loc2d = F.binary_cross_entropy_with_logits(
        scores2d.squeeze().masked_select(mask2d),
        labels2d.masked_select(mask2d)
    )
    return loss_loc2d

def lossfun_softloc(slogits, elogits, s_labels, e_labels, vmask, temperature):
    from models.SeqPANLib.layers import mask_logits
    slogits = mask_logits(slogits, vmask)
    elogits = mask_logits(elogits, vmask)
    s_labels = mask_logits(s_labels, vmask)
    e_labels = mask_logits(e_labels, vmask)
    
    slogits = F.softmax(F.normalize(slogits, p=2, dim=1) / temperature, dim=-1) 
    elogits = F.softmax(F.normalize(elogits, p=2, dim=1) / temperature, dim=-1) 
    s_labels = F.softmax(F.normalize(s_labels, p=2, dim=1) / temperature, dim=-1) 
    e_labels = F.softmax(F.normalize(e_labels, p=2, dim=1) / temperature, dim=-1) 


    # sloss = F.cross_entropy(slogits, s_labels, reduce="batchmean")
    # eloss = F.cross_entropy(elogits, e_labels, reduce="batchmean")
     
    sloss = torch.sum(F.kl_div(slogits.log(), s_labels, reduction='none'), dim=1)
    eloss = torch.sum(F.kl_div(elogits.log(), e_labels, reduction='none'), dim=1)

    return sloss + eloss