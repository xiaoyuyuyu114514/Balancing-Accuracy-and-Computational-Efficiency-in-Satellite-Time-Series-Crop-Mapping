"""E1 changes only loss; E2's boundary head is removed for deployment."""
import torch
from torch import nn
from reference_model import StateTrendUNet
from base_models import get_model


def base_model(family):
    return StateTrendUNet() if family=='ours' else get_model('utae')


class BoundaryModel(nn.Module):
    def __init__(self,family,variant):
        super().__init__();self.family=family;self.variant=variant
        self.base=base_model(family);self._feature=None;self._capture=False;self._hook=None
        self.boundary_head=None
        if variant in ['e2','e3']:
            # Preserve the reference base initialization/RNG state. The same small
            # head initialization is used for both model families.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261101)
                self.boundary_head=nn.Sequential(nn.Conv2d(32,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,1))
                for module in self.boundary_head.modules():
                    if isinstance(module,nn.Conv2d):
                        nn.init.xavier_normal_(module.weight);nn.init.zeros_(module.bias)
            classifier=self.base.spatial.head if family=='ours' else self.base.net.out_conv
            self._hook=classifier.register_forward_pre_hook(self._remember_feature)

    def _remember_feature(self,module,args):
        if self._capture:self._feature=args[0]

    def forward(self,x,batch_positions,valid_mask,with_aux=False):
        self._capture=bool(with_aux and self.boundary_head is not None)
        try:
            logits=self.base(x,batch_positions=batch_positions,valid_mask=valid_mask)
            aux=self.boundary_head(self._feature) if self._capture else None
            return logits,aux
        finally:
            self._feature=None;self._capture=False

    def deployment_state(self):
        return self.base.state_dict()
