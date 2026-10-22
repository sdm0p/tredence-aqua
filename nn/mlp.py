import numpy as np
from engine import Tensor,cross_entropy
class Linear:
    def __init__(self,inp,out,rng): self.w=Tensor(rng.normal(0,np.sqrt(2/inp),(inp,out)),True); self.b=Tensor(np.zeros(out),True); self.mask=np.ones((inp,out),bool)
    def __call__(self,x): return x@(self.w* self.mask)+self.b
    def params(self): return [self.w,self.b]
class MLP:
    def __init__(self,rng,d=2,h=32,k=3): self.layers=[Linear(d,h,rng),Linear(h,h,rng),Linear(h,k,rng)]
    def __call__(self,x):
        z=Tensor(x);z=self.layers[0](z).relu();z=self.layers[1](z).relu();return self.layers[2](z)
    def params(self): return [p for l in self.layers for p in l.params()]
    def weights(self): return [l.w for l in self.layers]
    def predict(self,x): return np.argmax(self(x).data,1)
    def dense_forward(self,x):
        z=np.asarray(x)
        for i,l in enumerate(self.layers):
            z=z@(l.w.data*l.mask)+l.b.data
            if i<len(self.layers)-1:z=np.maximum(z,0)
        return z
class Adam:
    def __init__(self,params,lr=2e-3):
        self.p=params;self.lr=lr;self.m=[np.zeros_like(x.data) for x in params];self.v=[np.zeros_like(x.data) for x in params];self.t=0
        self.age=[np.zeros_like(x.data,dtype=int) for x in params];self.masks=[np.ones_like(x.data,dtype=bool) for x in params]
    def set_mask(self,param,mask,revival_value=0.):
        i=self.p.index(param);mask=np.asarray(mask)
        if mask.shape!=param.data.shape or not np.all((mask==0)|(mask==1)):raise ValueError('invalid mask')
        mask=mask.astype(bool);changed=mask!=self.masks[i];param.data[changed&mask]=revival_value;param.data[~mask]=0
        for state in (self.m[i],self.v[i],self.age[i]):state[changed|~mask]=0
        if param.grad is not None:param.grad[changed|~mask]=0
        self.masks[i]=mask.copy()
    def step(self):
        self.t+=1
        for i,(x,m,v) in enumerate(zip(self.p,self.m,self.v)):
            if x.grad is None:continue
            mask=self.masks[i];g=np.where(mask,x.grad,0);self.age[i][mask]+=1
            m[:]=np.where(mask,.9*m+.1*g,0);v[:]=np.where(mask,.999*v+.001*g*g,0);age=np.maximum(self.age[i],1)
            x.data-=self.lr*(m/(1-.9**age))/(np.sqrt(v/(1-.999**age))+1e-8);x.data[~mask]=0;x.grad=None
