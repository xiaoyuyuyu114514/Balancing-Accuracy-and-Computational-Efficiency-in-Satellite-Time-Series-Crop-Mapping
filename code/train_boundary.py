import argparse,hashlib,json,random,time,os
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from boundary_models import BoundaryModel
from experiment import ROOT,save,state_hash,stage,evaluate_checkpoint
from fast_data import RawPastis,GPUNormalizer


def main(args):
    folder=Path(os.environ.get('RUNS_DIR', ROOT.parent/'runs'))/f'{args.family}_{args.variant}';folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(1);torch.cuda.manual_seed_all(1);np.random.seed(1);random.seed(1)
    torch.cuda.set_per_process_memory_fraction(.85)
    model=BoundaryModel(args.family,args.variant)
    initial_hash=state_hash(model.base.state_dict())
    model=model.cuda()
    datasets=[RawPastis([1,2,3]),RawPastis([4])];norms=[GPUNormalizer(d) for d in datasets]
    orders=np.load(ROOT/'epoch_orders.npy')
    config=dict(family=args.family,display_family='TP-EAN (StateTrendUNet)' if args.family=='ours' else 'U-TAE',
        variant=args.variant,epochs=100,batch_size=2,seed=1,optimizer='Adam',lr=.001,weight_decay=0.,
        schedule='constant',from_scratch=True,alpha=2.,boundary_radius=1,lambda_aux=.1,
        loss='weighted CE' if args.variant=='e1' else 'CE + .1*(BCE+Dice)' if args.variant=='e2' else 'weighted CE + .1*(BCE+Dice)',
        boundary='Both endpoints of unlike 4-connected valid semantic labels; background valid; label 19 ignored',
        distance='Euclidean pixel-center distance',boundary_tolerance=1,error_band_radius=2,
        inference_head_removed=True,boundary_never_fed_into_classification=True,
        train_parameters=sum(p.numel() for p in model.parameters()),inference_parameters=sum(p.numel() for p in model.base.parameters()),
        auxiliary_parameters=sum(p.numel() for p in model.parameters())-sum(p.numel() for p in model.base.parameters()),
        base_initial_sha256=initial_hash,sampling_sha256=hashlib.sha256((ROOT/'epoch_orders.npy').read_bytes()).hexdigest(),
        folds=dict(train=[1,2,3],val=[4],test=[5]),samples=[1455,482,496],image_size=128,full_time_series=True,
        normalization='same official per-split fold statistics as E0',precision='BF16 network; FP32 losses',
        checkpoint_selection='maximum validation mIoU, same primary convention as E0')
    config.update(loss='E1 weighted CE + alpha*Lovasz' if 'lovasz' in args.variant else 'E1 weighted CE',lovasz_alpha={'e1_lovasz02':.2,'e1_lovasz05':.5}.get(args.variant,0.),lovasz_classes='present',lovasz_per_image=False,temporal_dropout=.2 if args.variant=='e1_td20' else 0.,augmentation_train_only=True,validation_full_sequence=True,augmentation_seed='20261102 + epoch*10000 + step')
    save(folder/'config.json',config);print(json.dumps(config),flush=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    torch.cuda.reset_peak_memory_stats()
    if args.smoke:
        tr,_=stage(model,datasets[0],orders[0].tolist(),norms[0],folder,'smoke_train',0,args.variant,optimizer,limit=2)
        va,_=stage(model,datasets[1],list(range(len(datasets[1]))),norms[1],folder,'smoke_val',0,limit=1)
        save(folder/'smoke.json',dict(passed=True,train=tr,val=va,peak_GiB=torch.cuda.max_memory_allocated()/2**30));return
    history=[];first=1;best=-1.;prior=0.
    if (folder/'last.pt').exists():
        cp=torch.load(folder/'last.pt',weights_only=False)
        assert cp['config']['variant']==args.variant and cp['config']['family']==args.family
        model.load_state_dict(cp['state_dict']);optimizer.load_state_dict(cp['optimizer'])
        history=cp['history'];first=cp['epoch']+1;best=cp['best'];prior=cp['wall_seconds']
        torch.set_rng_state(cp['rng']);torch.cuda.set_rng_state_all(cp['cuda_rng'])
    start=time.perf_counter();save(folder/'status.json',dict(status='running',next_epoch=first,target_epochs=100,started=datetime.now().isoformat()))
    for epoch in range(first,101):
        tr,_=stage(model,datasets[0],orders[epoch-1].tolist(),norms[0],folder,'train',epoch,args.variant,optimizer)
        va,cm=stage(model,datasets[1],list(range(len(datasets[1]))),norms[1],folder,'val',epoch)
        history.append(dict(epoch=epoch,train=tr,val=va));save(folder/'history.json',history)
        if va['mIoU']>best:
            best=va['mIoU'];torch.save(dict(epoch=epoch,state_dict=model.state_dict(),val=va,config=config),folder/'best.tmp')
            (folder/'best.tmp').replace(folder/'best.pt')
        torch.save(dict(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),history=history,
                        best=best,wall_seconds=prior+time.perf_counter()-start,rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(),config=config),folder/'last.tmp')
        (folder/'last.tmp').replace(folder/'last.pt')
        print(f'EPOCH {epoch}/100 val mIoU={va["mIoU"]:.3f} BF1={va["boundary_f1"]:.3f}',flush=True)
    cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(cp['state_dict'])
    deploy={k:v.detach().cpu() for k,v in model.deployment_state().items()}
    torch.save(dict(state_dict=deploy,epoch=cp['epoch'],family=args.family,config=config,
                    boundary_head_removed=True),folder/'deployment.pt')
    del model,optimizer,cp
    import gc
    gc.collect();torch.cuda.empty_cache()
    epoch=torch.load(folder/'deployment.pt',map_location='cpu',weights_only=False)['epoch']
    outputs=evaluate_checkpoint(args.family,deploy,folder,epoch)
    assert len(history)==100 and all(r['train']['samples']==1455 and r['val']['samples']==482 for r in history)
    assert outputs['test']['samples']==496 and outputs['val']['samples']==482
    assert epoch==max(history,key=lambda r:r['val']['mIoU'])['epoch']
    result=dict(status='completed',family=args.family,variant=args.variant,epochs=100,best_epoch=epoch,
        val=outputs['val'],test=outputs['test'],performance=outputs['performance'],wall_seconds=prior+time.perf_counter()-start,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,ended=datetime.now().isoformat())
    save(folder/'verification.json',dict(verified=True,epochs=100,train_samples_each=1455,val_samples_each=482,
                                      test_samples=496,deployment_boundary_head_absent=True))
    save(folder/'completed.json',result);save(folder/'status.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['ours','utae'],required=True)
    p.add_argument('--variant',choices=['e1_lovasz02','e1_lovasz05','e1_td20'],required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    main(args)
