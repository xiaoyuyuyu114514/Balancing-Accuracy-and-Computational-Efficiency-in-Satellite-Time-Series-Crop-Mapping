import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

from src.backbones.utae import UTAE

def common_init(m):
    # Exactly the reference method's initialization for shared convolution types.
    # Linear layers use the same Xavier/zero-bias rule; norm defaults retained.
    if isinstance(m,(nn.Conv1d,nn.Conv2d,nn.Conv3d,nn.ConvTranspose2d,nn.ConvTranspose3d,nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:nn.init.zeros_(m.bias)


class CNNWrapper(nn.Module):
    def __init__(self,model):super().__init__();self.net=model
    def forward(self,x,batch_positions,valid_mask):return self.net(x,batch_positions=batch_positions)


def get_model(name):
    if name != "utae":
        raise ValueError(name)
    model = CNNWrapper(UTAE(input_dim=10, encoder_widths=[64,64,64,128],
        decoder_widths=[32,32,64,128], out_conv=[32,20], str_conv_k=4,
        str_conv_s=2, str_conv_p=1, agg_mode="att_group", encoder_norm="group",
        n_head=16, d_model=256, d_k=4, pad_value=0, padding_mode="reflect"))
    model.apply(common_init)
    return model