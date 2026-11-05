"""Single-factor E1 extensions; deployment architecture is unchanged."""
import torch
from fast_data import raw_collate
from lovasz_official import lovasz_softmax
from boundaries import objective as original_objective

VARIANTS={'e1_lovasz02':.2,'e1_lovasz05':.5,'e1_td20':0.}

def objective(logits,labels,variant,aux=None):
    base,parts=original_objective(logits,labels,'e1' if variant in VARIANTS else variant,aux)
    alpha=VARIANTS.get(variant,0.)
    lv=logits.float().sum()*0.
    if alpha and (labels!=19).any():
        with torch.autocast(device_type=logits.device.type,enabled=False):
            lv=lovasz_softmax(logits.float().softmax(1),labels,classes='present',per_image=False,ignore=19)
    parts['lovasz']=lv.detach()
    return base+alpha*lv,parts

def temporal_dropout(cpu_batch,epoch,step,probability=.2):
    # Local deterministic RNG: repeatable after resume; shared augmentation
    # draws for both families, independent of prefetch scheduling/model RNG.
    x,dates,valid,y=cpu_batch
    generator=torch.Generator().manual_seed(20261102+epoch*10000+step)
    batch=[]
    for i in range(len(y)):
        ids=valid[i].nonzero().flatten()
        keep=torch.rand(len(ids),generator=generator)>=probability
        if not keep.any():keep[torch.randint(len(ids),(1,),generator=generator)]=True
        ids=ids[keep]
        batch.append((x[i,ids],dates[i,ids],y[i]))
    return tuple(v.pin_memory() if x.is_pinned() else v for v in raw_collate(batch))
