"""Explicit semantic-boundary convention shared by losses and evaluation.

4-connected valid unlike labels mark BOTH adjacent pixels as boundary. Class 0
is valid; class 19 never creates an edge. Dilation/tolerance uses Euclidean pixel
center distance (disk radius 1 or 2), not a square/Chebyshev neighborhood.
"""
import torch
import torch.nn.functional as F

_KERNELS={}


@torch.no_grad()
def semantic_boundary(labels,valid):
    edge=torch.zeros_like(valid,dtype=torch.bool)
    horizontal=(labels[:,:,1:]!=labels[:,:,:-1])&valid[:,:,1:]&valid[:,:,:-1]
    vertical=(labels[:,1:,:]!=labels[:,:-1,:])&valid[:,1:,:]&valid[:,:-1,:]
    edge[:,:,1:]|=horizontal;edge[:,:,:-1]|=horizontal
    edge[:,1:,:]|=vertical;edge[:,:-1,:]|=vertical
    return edge


@torch.no_grad()
def dilate_disk(mask,radius):
    key=(mask.device.type,mask.device.index,radius)
    if key not in _KERNELS:
        axis=torch.arange(-radius,radius+1,device=mask.device)
        kernel=(axis[:,None].square()+axis[None,:].square()<=radius*radius).float()
        _KERNELS[key]=kernel[None,None]
    with torch.autocast(device_type=mask.device.type,enabled=False):
        return F.conv2d(mask[:,None].float(),_KERNELS[key],padding=radius)[:,0]>0


@torch.no_grad()
def targets(labels):
    valid=labels!=19
    edge=semantic_boundary(labels,valid)
    return dict(valid=valid,boundary=edge,band1=dilate_disk(edge,1)&valid,band2=dilate_disk(edge,2)&valid)


def objective(logits,labels,variant,aux_logits=None,alpha=2.,aux_weight=.1):
    masks=targets(labels);valid=masks['valid']
    with torch.autocast(device_type=logits.device.type,enabled=False):
        pixel_ce=F.cross_entropy(logits.float(),labels,ignore_index=19,reduction='none')
        weight=valid.float()
        if variant in ['e1','e3']:weight=weight*(1+alpha*masks['band1'].float())
        ce=(pixel_ce*weight).sum()/weight.sum().clamp_min(1.)
        zero=ce*0.;bce=zero;dice=zero
        if variant in ['e2','e3']:
            if aux_logits is None:raise ValueError('Auxiliary boundary logits required')
            z=aux_logits[:,0].float();target=masks['boundary'].float();vm=valid.float()
            bce=(F.binary_cross_entropy_with_logits(z,target,reduction='none')*vm).sum()/vm.sum().clamp_min(1.)
            probs=z.sigmoid()*vm;truth=target*vm
            numerator=2*(probs*truth).sum((1,2))+1e-6
            denominator=probs.sum((1,2))+truth.sum((1,2))+1e-6
            image_dice=1-numerator/denominator
            usable=(vm.sum((1,2))>0).float()
            dice=(image_dice*usable).sum()/usable.sum().clamp_min(1.)
        total=ce+aux_weight*(bce+dice)
    return total,dict(ce=ce.detach(),boundary_bce=bce.detach(),boundary_dice=dice.detach())


class BoundaryMetrics:
    def __init__(self,device):
        # pred boundary, gt boundary, matched pred, matched gt, near errors,
        # near pixels, interior errors, interior pixels, crop-interior errors/pixels.
        self.counts=torch.zeros(10,dtype=torch.int64,device=device)

    @torch.no_grad()
    def add(self,pred,gt):
        m=targets(gt);valid=m['valid'];truth=m['boundary']
        guess=semantic_boundary(pred,valid)
        near=m['band2'];inside=valid&~near;crop_inside=inside&(gt>0);error=pred!=gt
        values=[guess.sum(),truth.sum(),(guess&dilate_disk(truth,1)&valid).sum(),
                (truth&dilate_disk(guess,1)&valid).sum(),(error&near).sum(),near.sum(),
                (error&inside).sum(),inside.sum(),(error&crop_inside).sum(),crop_inside.sum()]
        self.counts+=torch.stack(values)

    def value(self):
        p,g,mp,mg,en,nn,ei,ni,ec,nc=self.counts.cpu().tolist()
        precision=mp/p if p else 0.;recall=mg/g if g else 0.
        f1=2*precision*recall/(precision+recall) if precision+recall else (1. if not p and not g else 0.)
        return dict(boundary_f1=100*f1,boundary_precision=100*precision,boundary_recall=100*recall,
            near_boundary_error_pct=100*en/nn if nn else None,
            interior_error_pct=100*ei/ni if ni else None,crop_interior_error_pct=100*ec/nc if nc else None,
            boundary_counts=dict(pred=p,gt=g,matched_pred=mp,matched_gt=mg),
            error_counts=dict(near_errors=en,near_pixels=nn,interior_errors=ei,interior_pixels=ni),
            boundary_tolerance_pixels=1,error_band_radius_pixels=2,distance='Euclidean',boundary_type='semantic, class-agnostic matching')
