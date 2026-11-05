"""Same split, normalization, and date ordering as the four-model experiment."""
import os, json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('PASTIS_ROOT', 'PASTIS'))


class Pastis(Dataset):
    def __init__(self, folds):
        meta = json.loads((DATA/'metadata.geojson').read_text())
        self.rows = sorted([f['properties'] for f in meta['features']
                            if f['properties']['Fold'] in folds], key=lambda x:int(x['ID_PATCH']))
        stats = json.loads((DATA/'NORM_S2_patch.json').read_text())
        self.mean = np.mean([stats[f'Fold_{f}']['mean'] for f in folds], axis=0).astype('float32')[None,:,None,None]
        self.std = np.mean([stats[f'Fold_{f}']['std'] for f in folds], axis=0).astype('float32')[None,:,None,None]
        ref = datetime(2018, 9, 1)
        self.dates = []
        for row in self.rows:
            ds = row['dates-S2']
            ds = json.loads(ds) if isinstance(ds, str) else ds
            vals = [(datetime.strptime(str(v), '%Y%m%d')-ref).days for v in ds.values()]
            assert vals == sorted(vals), 'Dates must match chronological image ordering'
            self.dates.append(torch.tensor(vals, dtype=torch.long))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        pid = int(self.rows[i]['ID_PATCH'])
        x = np.load(DATA/f'DATA_S2/S2_{pid}.npy').astype('float32')
        y = np.load(DATA/f'ANNOTATIONS/TARGET_{pid}.npy')[0].astype('int64')
        assert len(x) == len(self.dates[i])
        return torch.from_numpy((x-self.mean)/self.std), self.dates[i], torch.from_numpy(y)


def collate(batch):
    tmax = max(len(x) for x, _, _ in batch)
    b = len(batch)
    x = torch.zeros((b, tmax, *batch[0][0].shape[1:]))
    dates = torch.zeros((b, tmax), dtype=torch.long)
    valid = torch.zeros((b, tmax), dtype=torch.bool)
    for i, (xx, dd, _) in enumerate(batch):
        n = len(xx)
        x[i, :n] = xx
        dates[i, :n] = dd
        valid[i, :n] = True
    return x, dates, valid, torch.stack([y for _, _, y in batch])
