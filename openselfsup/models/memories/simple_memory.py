import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import get_dist_info

from openselfsup.utils import AliasMethod
from ..registry import MEMORIES


@MEMORIES.register_module
class SimpleMemory(nn.Module):

    def __init__(self, length, feat_dim, momentum, **kwargs):
        super(SimpleMemory, self).__init__()
        self.rank, self.num_replicas = get_dist_info()
        self.feature_bank = torch.randn(length, feat_dim).cuda()
        self.feature_bank = nn.functional.normalize(self.feature_bank)
        self.momentum = momentum
        self.multinomial = AliasMethod(torch.ones(length))
        self.multinomial.cuda()

    def update(self, ind, feature):
        feature_norm = nn.functional.normalize(feature)
        ind, feature_norm = self._gather(ind, feature_norm)
        feature_old = self.feature_bank[ind, ...]
        feature_new = (1 - self.momentum) * feature_old + \
                      self.momentum * feature_norm
        feature_new_norm = nn.functional.normalize(feature_new)
        self.feature_bank[ind, ...] = feature_new_norm

    def _gather(self, ind, feature):  # gather ind and feature
        ind_gathered = [
            torch.ones_like(ind).cuda() for _ in range(self.num_replicas)
        ]
        feature_gathered = [
            torch.ones_like(feature).cuda() for _ in range(self.num_replicas)
        ]
        dist.all_gather(ind_gathered, ind)
        dist.all_gather(feature_gathered, feature)
        ind_gathered = torch.cat(ind_gathered, dim=0)
        feature_gathered = torch.cat(feature_gathered, dim=0)
        return ind_gathered, feature_gathered
