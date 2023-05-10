Several methods was integrated into a single framework.

### Models Zoo

- SeqPAN:
    - https://github.com/26hzhang/SeqPAN.git
- SeqPANBackBone: 
    - remvoe the match loss and dual attention
- SeqPANBert:
    - replace the BERT word embedding to Glove.
- VSLNet
    - https://github.com/26hzhang/VSLNet.git
- BAN
    - https://github.com/DJX1995/BAN-APR.git
    - (still exsit bug, can run, but can't train )
- CCA
    - https://github.com/ZiyueWu59/CCA.git
- CPL
    - https://github.com/minghangz/cpl.git
    - (uncompleted)


**NOTE** 

There are many draft codes, we don't test the robustness. Especially the part of generating various labels in **BaseDataset.py** That easily produce bugs to decrease performance. Please test the consistency of label and vfeat length before traning.

<br>

### Video Feature
To download the video feature, please refer to https://github.com/IsaacChanghau/SeqPAN. It is possible to save the video feature in any preferred location.
Adjust "paths" value within the ./config/anet/SeqPAN.yaml.
<br>

## Quick Start
```python
# train
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/charades/SeqPAN.yaml 
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/anet/VSLNet.yaml

# debug
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/charades/SeqPAN.yaml --debug

# test accuracy
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/charades/SeqPAN.yaml --mode test --checkpoint ./ckpt/charades/best_SeqPAN.pkl

# test efficiency
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/charades/SeqPAN.yaml --mode summary

```