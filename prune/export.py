"""Export topology once; time only execution."""
import numpy as np
class SparseMLP:
    def __init__(self,model):
        self.layers=[];self.active=0
        for l in model.layers:
            cols=[]
            for j in range(len(l.b.data)):
                idx=np.flatnonzero(l.mask[:,j]);cols.append((idx,l.w.data[idx,j].copy()));self.active+=len(idx)
            self.layers.append((cols,l.b.data.copy()))
    def __call__(self,x):
        z=np.asarray(x)
        for i,(cols,b) in enumerate(self.layers):
            out=np.empty((len(z),len(b)))
            for j,(idx,w) in enumerate(cols):out[:,j]=z[:,idx]@w+b[j]
            z=np.maximum(out,0) if i<len(self.layers)-1 else out
        return z
