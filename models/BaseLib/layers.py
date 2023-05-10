import torch
import torch.nn as nn

## ------ mask ---------
def generate_2dmask(L, pooling_counts=None):
    if pooling_counts is None:
        pooling_counts = [L//4, L//8, L//8]

    mask2d = torch.zeros(L, L, dtype=torch.bool)
    mask2d[range(L), range(L)] = 1
    stride, offset = 1, 0
    for c in pooling_counts:
        for _ in range(c):
            # fill a diagonal line
            offset += stride
            i, j = range(0, L - offset), range(offset, L)
            mask2d[i, j] = 1
        stride *= 2
    return mask2d

def convert_length_to_mask(lengths, max_len):
    # lengths = torch.from_numpy(lengths)
    # max_len = lengths.max().item()
    mask = torch.arange(max_len).expand(lengths.size()[0], max_len) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    # return inputs * mask + mask_value * (1.0 - mask) 
    return inputs + mask_value * (1.0 - mask) 

## ------ mask ---------



class LinearBlock(nn.Module):
    def __init__(self, indim, outdim, droprate):
        super().__init__()
        self.linear_layer = nn.Linear(indim, outdim)
        self.layer_norm = nn.LayerNorm(outdim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        # x: [Batchsize, Len, Dim]
        x = self.linear_layer(x)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    