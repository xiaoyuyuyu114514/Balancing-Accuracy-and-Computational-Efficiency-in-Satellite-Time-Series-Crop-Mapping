"""Optional input pipeline: raw int16, bounded threaded prefetch, GPU normalization.

This module does not modify model code, checkpoints, or original training settings.
Threaded batches share the process address space, avoiding Windows worker IPC.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from data import DATA, Pastis


class RawPastis(Pastis):
    def __getitem__(self, i):
        pid = int(self.rows[i]['ID_PATCH'])
        x = np.load(DATA/f'DATA_S2/S2_{pid}.npy')
        if x.dtype != np.int16:
            raise TypeError(f'Expected int16 input, got {x.dtype}')
        y = np.load(DATA/f'ANNOTATIONS/TARGET_{pid}.npy')[0].astype('int64')
        assert len(x) == len(self.dates[i])
        return torch.from_numpy(x), self.dates[i], torch.from_numpy(y)


def raw_collate(batch):
    tmax = max(len(x) for x, _, _ in batch)
    b = len(batch)
    x = torch.zeros((b, tmax, *batch[0][0].shape[1:]), dtype=torch.int16)
    dates = torch.zeros((b,tmax), dtype=torch.long)
    valid = torch.zeros((b,tmax), dtype=torch.bool)
    for i, (xx,dd,_) in enumerate(batch):
        n = len(xx)
        x[i,:n] = xx
        dates[i,:n] = dd
        valid[i,:n] = True
    return x, dates, valid, torch.stack([y for _,_,y in batch])


class GPUNormalizer:
    def __init__(self, dataset, device='cuda'):
        self.mean = torch.as_tensor(dataset.mean, device=device)[None]
        self.std = torch.as_tensor(dataset.std, device=device)[None]
        self.device = device

    def __call__(self, cpu_batch):
        raw, dates, valid, y = [v.to(self.device, non_blocking=True) for v in cpu_batch]
        # FP32 normalization preserves the original int16 dynamic range. Do not
        # cast raw reflectances directly to BF16 before converting/normalizing.
        with torch.autocast(device_type='cuda', enabled=False):
            x = raw.float()
            x.sub_(self.mean).div_(self.std)
            x.masked_fill_(~valid[:,:,None,None,None], 0.)
        return x, dates, valid, y


class ThreadBatchLoader:
    """Finite ordered prefetch, no whole-dataset copies or unbounded caching.

    Each worker builds one complete CPU batch and pins it. Only the caller
    transfers tensors to the GPU. Caller supplies the desired sample order.
    """
    def __init__(self, dataset, indices, batch_size=2, workers=4, prefetch=4):
        self.dataset = dataset
        self.indices = list(indices)
        self.batch_size = batch_size
        self.workers = workers
        self.prefetch = prefetch

    def __len__(self):
        return (len(self.indices)+self.batch_size-1)//self.batch_size

    def _batch(self, ids):
        return tuple(v.pin_memory() for v in raw_collate([self.dataset[i] for i in ids]))

    def __iter__(self):
        chunks = iter([self.indices[i:i+self.batch_size]
                       for i in range(0,len(self.indices),self.batch_size)])
        pool = ThreadPoolExecutor(max_workers=self.workers)
        pending = deque()
        try:
            for _ in range(self.prefetch):
                ids = next(chunks, None)
                if ids is not None:
                    pending.append(pool.submit(self._batch, ids))
            while pending:
                ready = pending.popleft().result()
                ids = next(chunks, None)
                if ids is not None:
                    pending.append(pool.submit(self._batch, ids))
                yield ready
        finally:
            for task in pending:
                task.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
