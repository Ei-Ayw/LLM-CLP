"""
=============================================================================
IACD: Identity-Agnostic Contrastive Debiasing
File 2/3: DDP Utilities (GatherLayer + DDPStratifiedIdentitySampler)
=============================================================================
"""
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import Sampler


class GatherLayer(torch.autograd.Function):
    """支持反向传播的 all_gather, 解决原生 dist.all_gather 截断梯度的问题"""
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def diff_all_gather(x):
    """可微分的 all_gather: 单机直接返回, DDP 环境下走 GatherLayer"""
    if not dist.is_initialized():
        return x
    return torch.cat(GatherLayer.apply(x), dim=0)


class DDPStratifiedIdentitySampler(Sampler):
    """
    DDP 感知的分层采样器, 确保每个 batch 包含 4 象限样本:
      toxic+identity, toxic+no_identity, clean+identity, clean+no_identity
    每个 rank 获取不重叠的子集.
    """
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank

        y = dataset.y_tox
        has_id = dataset.has_id

        self.groups = {
            'toxic_id':   np.where((y >= 0.5) & (has_id == 1))[0],
            'toxic_noid': np.where((y >= 0.5) & (has_id == 0))[0],
            'clean_id':   np.where((y < 0.5) & (has_id == 1))[0],
            'clean_noid': np.where((y < 0.5) & (has_id == 0))[0],
        }

        self.n_per_group_total = batch_size // 4
        self.n_per_group_per_replica = max(1, self.n_per_group_total // self.num_replicas)

    def __iter__(self):
        indices = []
        shuffled = {k: np.random.permutation(v) for k, v in self.groups.items()}
        max_batches = min(len(v) // self.n_per_group_total for v in shuffled.values())

        for b in range(max_batches):
            batch = []
            for k in shuffled:
                start = b * self.n_per_group_total
                group_batch = shuffled[k][start:start + self.n_per_group_total]
                rank_start = self.rank * self.n_per_group_per_replica
                rank_end = rank_start + self.n_per_group_per_replica
                batch.extend(group_batch[rank_start:rank_end])

            np.random.shuffle(batch)
            indices.extend(batch)

        return iter(indices)

    def __len__(self):
        max_batches = min(len(v) // self.n_per_group_total for v in self.groups.values())
        return max_batches * (self.n_per_group_per_replica * 4)
