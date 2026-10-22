import numpy as np
def prune(model, target, saliency=True, grad_snapshots=None, optimizer=None):
    if not 0<=target<=1:raise ValueError('invalid target')
    ws=model.weights(); total=sum(w.data.size for w in ws); keep=total-int(np.floor(target*total+1e-10)); scores=[]
    for j,w in enumerate(ws):
        g = grad_snapshots[j] if grad_snapshots is not None else w.grad
        s=g if saliency and g is not None else np.abs(w.data)
        s=np.where(model.layers[j].mask,s,-np.inf); scores.append(s.ravel())
    flat=np.concatenate(scores); active=min(keep,sum(l.mask.sum() for l in model.layers)); keep_idx=np.argsort(-flat,kind='stable')[:active] if active else np.array([],int); chosen=np.zeros(total,bool);chosen[keep_idx]=True; pos=0
    for l,w in zip(model.layers,ws):
        n=w.data.size; l.mask=chosen[pos:pos+n].reshape(w.data.shape); w.data*=l.mask; pos+=n
        if w.grad is not None:w.grad*=l.mask
        if optimizer is not None:optimizer.set_mask(w,l.mask)
    return 1-sum(l.mask.sum() for l in model.layers)/total

def sparse_forward(model,x):
    z=np.asarray(x,dtype=float); macs=0
    for i,l in enumerate(model.layers):
        out=np.zeros((len(z),l.b.data.size))
        for j in range(l.b.data.size):
            idx=np.flatnonzero(l.mask[:,j]); out[:,j]=z[:,idx]@l.w.data[idx,j]+l.b.data[j]; macs+=len(idx)*len(z)
        z=out if i==len(model.layers)-1 else np.maximum(out,0)
    return z,macs
