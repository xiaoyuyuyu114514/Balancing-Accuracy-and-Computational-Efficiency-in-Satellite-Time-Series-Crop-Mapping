"""Shared training/evaluation utilities for semantic-boundary ablations."""
import hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from fast_data import RawPastis,ThreadBatchLoader,GPUNormalizer,raw_collate
from boundary_models import BoundaryModel,base_model
from boundaries import BoundaryMetrics
from extensions import objective,temporal_dropout

ROOT=Path(__file__).resolve().parent
def save(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2),encoding='utf-8');temp.replace(path)


def state_hash(state):
    h=hashlib.sha256()
    for key in sorted(state):
        tensor=state[key].detach().cpu().contiguous()
        h.update(key.encode());h.update(str(tensor.shape).encode());h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def segmentation_metrics(cm):
    raw=cm.double().cpu().numpy();c=raw.copy();c[19,:]=0;c[:,19]=0
    tp=c.diagonal();u=c.sum(0)+c.sum(1)-tp
    iou=np.divide(tp,u,out=np.full(20,np.nan),where=u>0)
    v=raw.copy();v[19,:]=0;tp2=v.diagonal()[:19];u2=v.sum(0)[:19]+v.sum(1)[:19]-tp2
    iou2=np.divide(tp2,u2,out=np.full(19,np.nan),where=u2>0)
    return dict(mIoU=float(np.nanmean(iou)*100),accuracy=float(tp.sum()/c.sum()*100),
        class_IoU=[None if np.isnan(x) else float(x*100) for x in iou],
        target_masked_mIoU=float(np.nanmean(iou2)*100),target_masked_accuracy=float(tp2.sum()/v.sum()*100),
        nonvoid_predicted_void=int(raw[:19,19].sum()))


def stage(model,dataset,indices,normalizer,folder,phase,epoch,variant='e0',optimizer=None,
          limit=None,save_predictions=False,progress_job=None):
    training=optimizer is not None;model.train(training)
    loader=ThreadBatchLoader(dataset,indices,batch_size=2,workers=4,prefetch=4)
    cm=torch.zeros((20,20),device='cuda',dtype=torch.int64)
    boundary=BoundaryMetrics('cuda') if not training else None
    totals=dict(loss=0.,ce=0.,boundary_bce=0.,boundary_dice=0.,lovasz=0.);samples=0;steps=0
    predictions=[]
    torch.cuda.synchronize();start=time.perf_counter()
    for step,cpu in enumerate(loader,1):
        if training and variant=='e1_td20':cpu=temporal_dropout(cpu,epoch,step)
        x,dates,valid,y=normalizer(cpu)
        if training:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training),torch.autocast('cuda',dtype=torch.bfloat16):
            if isinstance(model,BoundaryModel):logits,aux=model(x,dates,valid,with_aux=training)
            else:logits=model(x,batch_positions=dates,valid_mask=valid);aux=None
            # All validation losses are ordinary CE; segmentation metrics select checkpoints.
            loss,parts=objective(logits,y,variant if training else 'e0',aux)
        if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss {phase} epoch {epoch} step {step}')
        if training:loss.backward();optimizer.step();steps+=1
        pred=logits.detach().argmax(1)
        cm+=torch.bincount((y*20+pred).flatten(),minlength=400).reshape(20,20)
        if boundary:boundary.add(pred,y)
        if save_predictions:predictions.append(pred.cpu().numpy().astype('uint8'))
        totals['loss']+=loss.item()*len(y)
        for key,value in parts.items():totals[key]+=value.item()*len(y)
        samples+=len(y)
        if step==1 or step%50==0 or step==len(loader):
            progress=dict(model=progress_job or folder.name,epoch=epoch,stage=phase,step=step,steps=len(loader),
                          loss=totals['loss']/samples,seconds=time.perf_counter()-start,samples=samples)
            if progress_job:progress['family']=folder.name
            save(folder/'progress.json',progress)
            if progress_job:
                progress_root=ROOT/progress_job;progress_root.mkdir(exist_ok=True)
                save(progress_root/'progress.json',progress)
            print(json.dumps(progress),flush=True)
        if limit and step>=limit:break
    torch.cuda.synchronize()
    result=segmentation_metrics(cm)
    result.update({key:value/samples for key,value in totals.items()})
    result.update(seconds=time.perf_counter()-start,samples=samples,optimizer_steps=steps)
    if boundary:result.update(boundary.value())
    if save_predictions:
        array=np.concatenate(predictions);np.save(folder/f'predictions_{phase}.npy',array)
        save(folder/f'prediction_ids_{phase}.json',[int(dataset.rows[i]['ID_PATCH']) for i in indices[:samples]])
    return result,cm.cpu().numpy()


def deployment_performance(model,dataset,normalizer):
    """Same real B=1,T=46,128x128 input for all deployment measurements."""
    index=next(i for i,d in enumerate(dataset.dates) if len(d)==46)
    cpu=tuple(v.pin_memory() for v in raw_collate([dataset[index]]))
    x,dates,valid,_=normalizer(cpu);model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(5):model(x,batch_positions=dates,valid_mask=valid)
        torch.cuda.synchronize();times=[]
        for _ in range(30):
            start=time.perf_counter();model(x,batch_positions=dates,valid_mask=valid)
            torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    transpose_flops=[0]
    def count_transpose(module,inputs,output):
        z=inputs[0];transpose_flops[0]+=2*z.shape[0]*z.shape[2]*z.shape[3]*module.in_channels*(module.out_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
    handles=[m.register_forward_hook(count_transpose) for m in model.modules() if isinstance(m,nn.ConvTranspose2d)]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16),torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True,record_shapes=True) as prof:
        model(x,batch_positions=dates,valid_mask=valid)
    for handle in handles:handle.remove()
    counted=sum(e.flops for e in prof.key_averages())
    # PyTorch's supported FLOP annotation does not count ConvTranspose2d.
    if not any('conv_transpose' in e.key and e.flops for e in prof.key_averages()):counted+=transpose_flops[0]
    return dict(inference_parameters=sum(p.numel() for p in model.parameters()),
        latency_ms_median=statistics.median(times),latency_ms_p95=float(np.percentile(times,95)),
        gflops_counted=counted/1e9,input_shape=list(x.shape),patch_id=int(dataset.rows[index]['ID_PATCH']),
        warmup=5,repeats=30,includes_data_loading=False,includes_boundary_head=False,
        flops_convention='2 FLOPs per MAC; PyTorch supported-op count plus ConvTranspose2d MAC formula. Unsupported elementwise/normalization operators are not fully counted; this is an estimate, not an exhaustive hardware instruction count.')


def evaluate_checkpoint(family,state,folder,epoch,progress_job=None):
    model=base_model(family).cuda();model.load_state_dict(state,strict=True);model.eval()
    assert not model._forward_hooks
    datasets=[RawPastis([4]),RawPastis([5])];norms=[GPUNormalizer(d) for d in datasets]
    outputs={}
    for phase,ds,norm in zip(['val','test'],datasets,norms):
        metric,cm=stage(model,ds,list(range(len(ds))),norm,folder,phase,epoch,
                        save_predictions=True,progress_job=progress_job)
        metric['best_epoch']=epoch;save(folder/f'{phase}_metrics.json',metric)
        np.save(folder/f'confusion_{phase}.npy',cm);outputs[phase]=metric
    outputs['performance']=deployment_performance(model,datasets[1],norms[1])
    save(folder/'performance.json',outputs['performance'])
    return outputs
