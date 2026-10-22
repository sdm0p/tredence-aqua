"""Rebuild all evidence using held-out data and equivalent inference paths."""
import csv,json,platform,time,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train.run import train,spiral
from prune.export import SparseMLP
from engine import Tensor,cross_entropy

def write_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    out=Path(__file__).resolve().parent/'results';out.mkdir(exist_ok=True)
    rows=[];histories=[];timings=[]
    vx,vy=spiral(450,8);tx,ty=spiral(900,9)
    for method in ['magnitude','saliency']:
        for target in [0,.5,.75,.9,.95]:
            for seed in [0,1,2]:
                model,h=train(seed,80,target,method)
                logits=model.dense_forward(tx);val=model.dense_forward(vx)
                sparse=SparseMLP(model);parity=float(np.max(np.abs(logits-sparse(tx))))
                total=sum(l.mask.size for l in model.layers);active=sum(int(l.mask.sum()) for l in model.layers)
                row=dict(method=method,target_sparsity=target,seed=seed,achieved_sparsity=1-active/total,test_accuracy=float((logits.argmax(1)==ty).mean()),test_loss=float(cross_entropy(Tensor(logits),ty).data),validation_accuracy=float((val.argmax(1)==vy).mean()),train_accuracy=h[-1][1],active=active,total=total,biases=sum(l.b.data.size for l in model.layers),parity_error=parity,layer_active=json.dumps([int(l.mask.sum()) for l in model.layers]))
                rows.append(row)
                histories.extend(dict(method=method,target_sparsity=target,seed=seed,epoch=i+1,loss=a,train_accuracy=b) for i,(a,b) in enumerate(h))
                if target==.9:
                    for name,fn in [('dense',model.dense_forward),('sparse',sparse)]:
                        for _ in range(5):fn(tx)
                        for repeat in range(20):
                            start=time.perf_counter();fn(tx);ms=(time.perf_counter()-start)*1000
                            timings.append(dict(method=method,seed=seed,path=name,repeat=repeat,batch=len(tx),milliseconds=ms,weight_macs=len(tx)*(total if name=='dense' else active)))
    write_csv(out/'sweep.csv',rows);write_csv(out/'history.csv',histories);write_csv(out/'timings.csv',timings)
    metadata=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),processor=platform.processor(),epochs=80,data_seeds=dict(train=7,validation=8,test=9),sizes=[900,450,900],architecture=[2,32,32,3],lr=.002,ema=.9,seeds=[0,1,2],threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    (out/'environment.json').write_text(json.dumps(metadata,indent=2))
    plt.figure()
    for method in ['magnitude','saliency']:
        groups=[[r for r in rows if r['method']==method and r['target_sparsity']==s] for s in [0,.5,.75,.9,.95]]
        xs=[g[0]['achieved_sparsity'] for g in groups];ys=[np.mean([r['test_accuracy'] for r in g]) for g in groups];sd=[np.std([r['test_accuracy'] for r in g],ddof=1) for g in groups]
        plt.errorbar(xs,ys,yerr=sd,marker='o',label=method)
        for g in groups:plt.scatter([r['achieved_sparsity'] for r in g],[r['test_accuracy'] for r in g],s=10)
    plt.xlabel('Achieved weight sparsity');plt.ylabel('Held-out test accuracy');plt.legend();plt.tight_layout();plt.savefig(out/'pareto.png');plt.close()
    fig,axes=plt.subplots(1,2,figsize=(9,3))
    for target in [0,.9]:
        h=[r for r in histories if r['method']=='saliency' and r['seed']==0 and r['target_sparsity']==target]
        axes[0].plot([r['loss'] for r in h],label=str(target));axes[1].plot([r['train_accuracy'] for r in h],label=str(target))
    axes[0].set_title('Epoch mean minibatch loss');axes[1].set_title('Training accuracy')
    for ax in axes:ax.set_xlabel('Epoch');ax.legend()
    fig.tight_layout();fig.savefig(out/'training_curves.png');plt.close(fig)
    summary=['# Measured results','', 'Reproduced with `python evaluate_submission.py`. Accuracy below is on independent test data (seed 9). Three paired initialization/order seeds; error bars are sample standard deviation.','', '| Target | Achieved | Magnitude mean ± SD | Saliency mean ± SD |','|---|---|---|---|']
    for target in [0,.5,.75,.9,.95]:
        groups=[[r for r in rows if r['method']==m and r['target_sparsity']==target] for m in ['magnitude','saliency']]
        values=[np.array([r['test_accuracy'] for r in g])*100 for g in groups]
        summary.append(f'| {target:.0%} | {groups[0][0]["achieved_sparsity"]:.3%} | {values[0].mean():.2f} ± {values[0].std(ddof=1):.2f}% | {values[1].mean():.2f} ± {values[1].std(ddof=1):.2f}% |')
    summary+=['','## Falsifiable claim','']
    a=[r['test_accuracy'] for r in rows if r['method']=='saliency' and r['target_sparsity']==.9];b=[r['test_accuracy'] for r in rows if r['method']=='magnitude' and r['target_sparsity']==.9]
    summary.append(f'At target 90%, saliency averages {np.mean(a)*100:.2f}% held-out accuracy versus {np.mean(b)*100:.2f}% for magnitude across three paired seeds; paired difference {np.mean(np.array(a)-b)*100:.2f} percentage points. This is a small synthetic-task observation, not statistical proof of general superiority.')
    summary+=['','## Inference cost','']
    for name in ['dense','sparse']:
        v=[r for r in timings if r['path']==name];summary.append(f'{name}: median {np.median([r["milliseconds"] for r in v]):.4f} ms; {v[0]["weight_macs"]} weight MACs per batch of 900. Bias and activation work remains dense.')
    summary+=['',f'Maximum sparse/dense absolute error: {max(r["parity_error"] for r in rows):.3g}. Export and graph construction excluded from both timed inference paths. Raw repetitions: results/timings.csv. Timings are machine-dependent.','', 'Training remains dense masked computation. No sparse training speedup or production capacity claim is made. Data noise varies independently across splits, but all splits share the same synthetic task. No automatic regrowth is evaluated.']
    Path('RESULTS.md').write_text('\n'.join(summary),encoding='utf-8')
    print('\n'.join(summary))

if __name__=='__main__':main()
