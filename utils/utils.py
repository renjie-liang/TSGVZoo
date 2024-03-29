import json
import time
import torch
import numpy as np
import random
import pickle
import logging
import os
import yaml
import math
import h5py

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_json(filename):
    with open(filename, encoding='utf8') as fr:
        return json.load(fr)

def save_json(data, filename):
    with open(filename, "w", encoding='utf8') as fr:
        json.dump(data, fr)


def load_yaml(filename):
    with open(filename, encoding='utf8') as fr:
        return yaml.safe_load(fr)

def load_pickle(filename):
    with open(filename, mode='rb') as handle:
        data = pickle.load(handle)
        return data

def save_pickle(data, filename):
    with open(filename, mode='wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def time_idx(t, duration, vlen):
    if isinstance(t, list):
        res = []
        for i in t:
            res.append(time_idx(i, duration, vlen))
        return res
    else:
        return round(t / duration * (vlen - 1))

def frac_idx(frac, vlen):
    if isinstance(frac, list):
        res = []
        for i in frac:
            res.append(frac_idx(i, vlen))
        return res
    else:
        return round(frac * (vlen - 1))


def idx_time(t, duration, vlen):
    if isinstance(t, list):
        res = []
        for i in t:
            res.append(idx_time(i, duration, vlen))
        return res
    else:
        return round(t / (vlen-1) * duration, 2)

def set_seed_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_optimizer_and_scheduler(model, configs):
    from transformers import get_linear_schedule_with_warmup
    no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=configs.train.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, configs.train.num_train_steps * configs.train.warmup_proportion,
                                                configs.train.num_train_steps)
    return optimizer, scheduler


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()
    return apply_to_sample(_move_to_cuda, sample)

def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)




def get_logger(dir, tile):
    os.makedirs(dir, exist_ok=True)
    log_file = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(dir, "{}_{}.log".format(log_file, tile))

    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(levelname)s:%(message)s"
    # DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)

    fhlr = logging.FileHandler(log_file) 
    fhlr.setFormatter(formatter)
    fhlr.setLevel('INFO') 

    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger

BEST_ACC = 0
def save_best_model(score, model, save_name):
    global BEST_ACC
    if score > BEST_ACC:
        BEST_ACC = score
        torch.save(model.state_dict(), save_name)
        print("INFO: Save checkpoint to {}, ACC: {:.2f}".format(save_name, score))
    return BEST_ACC




# def plot_labels(s_labels, e_labels, m_labels, label_type):
#     from matplotlib import pyplot as plt
#     import numpy as np

#     if label_type == "VSL":
#         for i in range(s_labels.shape[0]):
#             plt.axvline(s_labels[i],  c='g', label="s_label")
#             plt.axvline(e_labels[i],  c='b', label="e_label")
#             # plt.plot(m_labels[i], )
#             plt.scatter(np.arange(m_labels.shape[1]), m_labels[i], c='y', label="h_label")

#             save_path = "./imgs/VSL_label/{}.jpg".format(i)
#             plt.legend()
#             print(save_path)
#             plt.savefig(save_path, dpi=300)
#             plt.cla()

#     elif label_type == "SeqPAN":
#         for i in range(s_labels.shape[0]):
#             plt.plot(s_labels[i], c='g', label="s_label")
#             plt.plot(e_labels[i], c='b', label="e_label")
#             plt.scatter(np.arange(m_labels.shape[1]), m_labels[i],  c='y', label="h_label")
#             plt.legend()
#             save_path = "./imgs/SeqPAN_label/{}.jpg".format(i)
#             print(save_path)
#             plt.savefig(save_path, dpi=300)
#             plt.cla()



# def iou_batch(i0, i1):
#     s = torch.stack([i0[0,:], i1[0,:]])
#     e = torch.stack([i0[1,:], i1[1,:]])
#     union = torch.stack([torch.min(s, dim=0)[0], torch.max(e, dim=0)[0]])
#     inter = torch.stack([torch.max(s, dim=0)[0], torch.min(e, dim=0)[0]])

#     iou = (inter[1,:] - inter[0,:]) / (union[1,:] - union[0,:])
#     iou = torch.clamp(iou, min=0.0, max=1.0)
#     return iou



# def get_gaussian_weight(center, vlen, L, alpha):
#     x = np.linspace(-1, 1, num=L,  dtype=np.float32)
#     sig = vlen / L
#     sig *= alpha
#     u = (center / L) * 2 - 1
#     weight = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
#     weight /= np.max(weight)
#     weight[vlen:] = 0.0
#     return weight

# def gene_soft_label(sidx, eidx, vlen, L, alpha):
#     Ssoft = get_gaussian_weight(sidx, vlen, L, alpha)
#     Esoft = get_gaussian_weight(eidx, vlen, L, alpha)
    
#     # O, Start, Internel, End
#     IOsoft = 1 - Ssoft - Esoft
#     mask_I = np.zeros(L)
#     mask_I[sidx:eidx+1] = 1
#     Isoft = IOsoft * mask_I

#     mask_O = np.zeros(L)
#     mask_O[:sidx] = 1
#     mask_O[eidx+1:vlen] = 1
#     Osoft = IOsoft * mask_O

#     Msoft = np.stack([Osoft, Ssoft, Isoft, Esoft]).T
    
#     return Ssoft, Esoft, Msoft










## ---------------- evaluate accuracy -------------
def iou_n1(candidates, gt):
    '''
    candidates: (prop_num, 2)
    gt: (2, )
    '''
    start, end = candidates[:, 0], candidates[:, 1]
    s, e = gt[0].float(), gt[1].float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    if (union[1] - union[0]) == 0.0:
        return 0.0
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)

def append_ious(ious, se_gts, se_props):
    for i in range(len(se_gts)):
        gt_se = se_gts[i]
        prop_se = se_props[i]
        iou = calculate_iou(gt_se, prop_se)
        ious.append(iou)
    return ious

def calculate_diou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    if (union[1] - union[0]) == 0.0:
        return 0.0
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    iou = iou * (1 - abs(i0[0]-i1[0])) * (1 - abs(i0[1]-i1[1]))
    return max(0.0, iou)

def append_dious(dious, se_gts, se_props):
    for i in range(len(se_gts)):
        gt_se = se_gts[i]
        prop_se = se_props[i]
        iou = calculate_diou(gt_se, prop_se)
        dious.append(iou)
    return dious

def calculate_iou_accuracy(ious, threshold):
    total_size = float(len(ious))
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return float(count) / total_size * 100.0

def get_i345_mi(ious):
    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    # mi = torch.mean(ious) * 100.0
    mi = np.mean(ious) * 100.0
    return r1i3, r1i5, r1i7, mi

## ---------------- evaluate accuracy -------------
