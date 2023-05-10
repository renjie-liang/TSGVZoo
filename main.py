from distutils.command.config import config
import os
import argparse
import torch
from torch import nn

import numpy as np
from easydict import EasyDict
from tqdm import tqdm

# from models.loss import append_ious, get_i345_mi
from utils.data_gen import load_dataset
from utils.data_utils import VideoFeatureDict
from utils.utils import load_json, load_yaml, AverageMeter, get_logger
from utils.utils import set_seed_config, build_optimizer_and_scheduler, save_best_model
from utils.DataLoader import get_loader
from models import *
import yaml, json
from utils.engine import train_epoch, test_epoch

torch.set_printoptions(precision=4, sci_mode=False)
def build_load_model(configs, args, word_vector):
    model = eval(configs.model.name)(configs, word_vector)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model  = model.to(configs.device)
    if args.checkpoint:
        model_checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(model_checkpoint)
    # for m in model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.xavier_uniform_(m.weight)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, required=True, help='config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--debug', action='store_true', help='only debug')
    parser.add_argument('--note', type=str, default='', help='task note')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    return parser.parse_args()

args = parse_args()
configs = EasyDict(load_yaml(args.config))
device = ("cuda" if torch.cuda.is_available() else "cpu" )
configs.device = device

set_seed_config(args.seed)
dataset = load_dataset(configs)
configs.num_chars = dataset['n_chars']
configs.num_words = dataset['n_words']

# get train and test loader
visual_features = VideoFeatureDict(configs.paths.feature_path, configs.model.max_vlen, args.debug)
train_loader = get_loader(dataset['train_set'], visual_features, configs, loadertype="train")
test_loader = get_loader(dataset['test_set'], visual_features, configs, loadertype="test")
# train_nosuffle_loader = get_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs, loadertype="test")
configs.train.num_train_steps = len(train_loader) * configs.train.epochs

# init logger,  meter, and checkpoint director
ckpt_dir = os.path.join(configs.paths.ckpt_dir, configs.task)
os.makedirs(ckpt_dir, exist_ok=True)
save_name = os.path.join(ckpt_dir, "best_{}.pkl".format(configs.model.name))

log_path = os.path.join(configs.paths.logs_dir, configs.task)
logger = get_logger(log_path, configs.model.name)
logger.info(args)
logger.info(json.dumps(configs, indent=4))
lossmeter = AverageMeter()

# init function about training, inference
infer_fun = eval("infer_" + configs.model.name)
train_engine = eval("train_engine_" + configs.model.name)
 
if args.mode == "train":
    # build model
    model = build_load_model(configs, args, dataset['word_vector'])
    optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
    best1, best2, best1_line, best2_line = 0, 0, "", ""
    # best_r1i7, best_miou, global_step = -1.0, -1.0, 0
    for epoch in range(configs.train.epochs):
        logger.info("Epoch {}|{}:".format(epoch, configs.train.epochs))
        
        ri, dri, loss = train_epoch(model, train_loader, configs, optimizer, scheduler, infer_fun, train_engine)
        train_line = "TRAIN:\tR3:{:.2f}\tR5:{:.2f}\tR7:{:.2f}\tmIoU:{:.2f}\tloss:{:.4f}".format(ri[0], ri[1], ri[2], ri[3], loss)
        logger.info(train_line)
        
        tri, tdri, tloss = test_epoch(model, test_loader, configs, infer_fun, train_engine)
        test_line  = "TEST: \tR3:{:.2f}\tR5:{:.2f}\tR7:{:.2f}\tmIoU:{:.2f}\tloss:{:.4f}".format(tri[0], tri[1], tri[2], tri[3], tloss)
        logger.info(test_line)

        if tri[2] > best1:
            best1 = tri[2]
            best1_line = "\n" + train_line + "\n" + test_line 
            torch.save(model.state_dict(), save_name)
        if tri[3] > best2:
            best2 = tri[3]
            best2_line = "\n" +  train_line + "\n" + test_line 
            
    logger.info("\n\nR1i7 in IID")
    logger.info(best1_line)
    logger.info("\nR1i7 in OOD")
    logger.info(best2_line)



elif args.mode == "test":
    model = build_load_model(configs, args, dataset['word_vector'])
    model.eval()
    
    tri, tdri, tloss = test_epoch(model, test_loader, configs, infer_fun, train_engine)
    test_line  = "TEST: \tR3:{:.2f}\tR5:{:.2f}\tR7:{:.2f}\tmIoU:{:.2f}\tloss{:.4f}".format(tri[0], tri[1], tri[2], tri[3], tloss)
    logger.info(test_line)



elif args.mode == "summary":
    configs.train.batch_size = 1
    test_loader = get_loader(dataset['test_set'], visual_features, configs, loadertype="test")
    
    import time 
    from fvcore.nn import FlopCountAnalysis
    model = build_load_model(configs, args, dataset['word_vector'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters() )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    count, total_time = 0, 0
    for data, _ in tqdm(test_loader):
        data = {key: value.to(configs.device) for key, value in data.items()}
        input_data = (data['words_ids'], data['char_ids'], data['vfeats'], data['vmasks'], data['tmasks'])

        if count == 0:
            flops = FlopCountAnalysis(model, input_data)
        start_time = time.time()
        model(*input_data)
        during_time = time.time() - start_time
        total_time += during_time
        count += 1
        average_time = total_time / count 
        # print("average time: ", average_time)
        # break
    
    print("FLOPS:", flops.total() / 1000000000) # M
    print("total Params:", total_params / 1000000) # M
    print("Params:", trainable_params / 1000000) # M
    print("Times:", average_time * 1000) # ms

print("Done!")