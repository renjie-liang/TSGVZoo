from distutils.command.config import config
import os
import argparse
import torch
from torch import nn

import numpy as np
from easydict import EasyDict
from tqdm import tqdm

from models.loss import append_ious, get_i345_mi
from utils.data_gen import load_dataset
from utils.data_utils import load_video_features
from utils.utils import load_json, set_seed_config, build_optimizer_and_scheduler, plot_labels, AverageMeter, get_logger, save_best_model
from utils.utils import build_load_model
from utils.data_loader import get_loader
from utils.engine import train_engine_SeqPAN, infer_SeqPAN, train_engine_CPL, infer_CPL
torch.set_printoptions(precision=4, sci_mode=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, required=True, help='config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--eval', action='store_true', help='only evaluate')
    parser.add_argument('--suffix', type=str, default='', help='task suffix')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    return parser.parse_args()


args = parse_args()
configs = EasyDict(load_json(args.config))
configs['suffix'] = args.suffix

set_seed_config(args.seed)
dataset = load_dataset(configs)
configs.num_chars = dataset['n_chars']
configs.num_words = dataset['n_words']

# get train and test loader
visual_features = load_video_features(configs.paths.feature_path, configs.model.vlen)
train_loader = get_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs, loadertype="train")
test_loader = get_loader(dataset=dataset['test_set'], video_features=visual_features, configs=configs, loadertype="test")
# train_nosuffle_loader = get_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs, loadertype="test")
configs.train.num_train_steps = len(train_loader) * configs.train.epochs


ckpt_dir = os.path.join(configs.paths.ckpt_dir, "{}_{}".format(configs.task, configs.suffix))
os.makedirs(ckpt_dir, exist_ok=True)
device = ("cuda" if torch.cuda.is_available() else "cpu" )
configs.device = device



# init logger and meter
logger = get_logger(ckpt_dir, "eval")
logger.info(args)
logger.info(configs)
lossmeter = AverageMeter()

# train and test
if not args.eval:
    # build model
    model = build_load_model(configs, args, dataset['word_vector'])
    # for m in model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.xavier_uniform_(m.weight)

    optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
    best_r1i7, global_step, mi_val_best = -1.0, 0, 0
    for epoch in range(configs.train.epochs):
        model.train()
        lossmeter.reset()
        tbar, ious = tqdm(train_loader), []
        for data in tbar:
            records = data[0]
            train_engine = eval("train_engine_" + configs.model.name)
            loss, output = train_engine(model, data, configs)


            lossmeter.update(loss.item())
            tbar.set_description("TRAIN {:2d}|{:2d} LOSS:{:.4f}".format(epoch + 1, configs.train.epochs, lossmeter.avg))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), configs.train.clip_norm)  # clip gradient
            optimizer.step()
            scheduler.step()


            infer_fun = eval("infer_" + configs.model.name)
            start_fracs, end_fracs = infer_fun(output, configs)
            ious = append_ious(ious, records, start_fracs, end_fracs)
        r1i3, r1i5, r1i5, r1i7, mi = get_i345_mi(ious)
        logger.info("TRAIN|\tR1I3: {:.2f}\tR1I5: {:.2f}\tR1I7: {:.2f}\tmIoU: {:.2f}\tloss:{:.4f}".format(mi, r1i3, r1i5, r1i7, lossmeter.avg))

    
        model.eval()
        lossmeter.reset()
        tbar = tqdm(test_loader)
        ious, ious_my = [], []

        for data in tbar:
            records = data[0]
            train_engine = eval("train_engine_" + configs.model.name)
            loss, output = train_engine(model, data, configs)
            
            lossmeter.update(loss.item())
            tbar.set_description("TEST  {:2d}|{:2d} LOSS:{:.4f}".format(epoch + 1, configs.train.epochs, lossmeter.avg))
            
            infer_fun = eval("infer_" + configs.model.name)
            start_fracs, end_fracs = infer_fun(output, configs)
            ious = append_ious(ious, records, start_fracs, end_fracs)

        r1i3, r1i5, r1i5, r1i7, mi = get_i345_mi(ious)
        save_name = os.path.join(ckpt_dir, "best.pkl")
        save_best_model(mi, model, save_name)

        logger.info("TEST |\tR1I3: {:.2f}\tR1I5: {:.2f}\tR1I7: {:.2f}\tmIoU: {:.2f}\tloss:{:.4f}".format(mi, r1i3, r1i5, r1i7, lossmeter.avg))
        logger.info("")

if args.eval:
    model = build_load_model(configs, dataset['word_vector'])
    model.eval()
    lossmeter.reset()
    tbar = tqdm(test_loader)
    ious = []
    for data in tbar:
        records = data[0]
        train_engine = eval("train_engine_" + configs.model.name)
        loss, output = train_engine(model, data, configs)
        lossmeter.update(loss.item())
        infer_fun = eval("infer_" + configs.model.name)
        start_fracs, end_fracs = infer_fun(output, configs)
        ious = append_ious(ious, records, start_fracs, end_fracs)
    r1i3, r1i5, r1i5, r1i7, mi = get_i345_mi(ious)
    logger.info("TEST |\tR1I3: {:.2f}\tR1I5: {:.2f}\tR1I7: {:.2f}\tmIoU: {:.2f}\tloss:{:.4f}".format(mi, r1i3, r1i5, r1i7, lossmeter.avg))
    logger.info("")

print("Done!")