import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils.utils import append_ious, get_i345_mi, append_dious, AverageMeter
from tqdm import tqdm


def train_epoch(model, dataloader, configs, optimizer, scheduler, infer_fun, train_engine):
    model.train()
    lossmeter, ious, dious = AverageMeter(), [], []
    for data in tqdm(dataloader):
        inputbatch, records = data
        loss, output = train_engine(model, inputbatch, configs, "train")
        lossmeter.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), configs.train.clip_norm)
        optimizer.step()
        scheduler.step()

        props_frac = infer_fun(output, configs)
        ious = append_ious(ious,  inputbatch["se_fracs"], props_frac)
        dious = append_dious(dious,  inputbatch["se_fracs"], props_frac)
    ri = get_i345_mi(ious)
    dri = get_i345_mi(dious)
    return ri, dri, lossmeter.avg


def test_epoch(model, dataloader, configs, infer_fun, train_engine):
    model.eval()
    lossmeter, ious, dious = AverageMeter(), [], []
    for data in tqdm(dataloader):
        inputbatch, records = data
        loss, output = train_engine(model, inputbatch, configs, "train")
        lossmeter.update(loss.item())
        props_frac = infer_fun(output, configs)
        ious = append_ious(ious,  inputbatch["se_fracs"], props_frac)
        dious = append_dious(dious,  inputbatch["se_fracs"], props_frac)
    ri = get_i345_mi(ious)
    dri = get_i345_mi(dious)
    return ri, dri, lossmeter.avg



