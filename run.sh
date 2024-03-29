# charades
CUDA_VISIBLE_DEVICES=-1 python main.py --config ./config/charades/SeqPAN.yaml
CUDA_VISIBLE_DEVICES=-1 python main.py --config ./config/charades/VSLNet.yaml
CUDA_VISIBLE_DEVICES=-1 python main.py --config ./config/charades/BAN.yaml

# anet
CUDA_VISIBLE_DEVICES=-1 python main.py --config ./config/anet/SeqPAN.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/anet/BAN.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/anet/CCA.yaml

# tacos
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/tacos/SeqPAN.yaml
