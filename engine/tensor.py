import numpy as np

def _unbroadcast(g, shape):
    while g.ndim > len(shape): g = g.sum(axis=0)
    for i,s in enumerate(shape):
        if s == 1 and g.shape[i] != 1: g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)

class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data=np.asarray(data,dtype=np.float64); self.requires_grad=requires_grad; self.grad=None
        self._prev=set(_children); self._backward=lambda:None; self._op=_op
    def __add__(self,o): return self._bin(o,lambda a,b:a+b,lambda g,a,b:(g,g))
    def __radd__(self,o): return self+o
    def __sub__(self,o): return self._bin(o,lambda a,b:a-b,lambda g,a,b:(g,-g))
    def __rsub__(self,o): return Tensor(o)-self
    def __mul__(self,o): return self._bin(o,lambda a,b:a*b,lambda g,a,b:(g*b,g*a))
    def __rmul__(self,o): return self*o
    def __truediv__(self,o): return self._bin(o,lambda a,b:a/b,lambda g,a,b:(g/b,-g*a/(b*b)))
    def __neg__(self): return self*-1
    def _bin(self,o,fn,der):
        o=o if isinstance(o,Tensor) else Tensor(o); out=Tensor(fn(self.data,o.data),self.requires_grad or o.requires_grad,(self,o))
        def bw():
            ga,gb=der(out.grad,self.data,o.data)
            if self.requires_grad:self._add_grad(_unbroadcast(ga,self.data.shape))
            if o.requires_grad:o._add_grad(_unbroadcast(gb,o.data.shape))
        out._backward=bw; return out
    def __matmul__(self,o):
        o=o if isinstance(o,Tensor) else Tensor(o)
        if self.data.ndim!=2 or o.data.ndim!=2:raise ValueError('matmul requires 2D operands')
        out=Tensor(self.data@o.data,self.requires_grad or o.requires_grad,(self,o))
        def bw():
            if self.requires_grad:self._add_grad(out.grad@np.swapaxes(o.data,-1,-2))
            if o.requires_grad: o._add_grad(np.swapaxes(self.data,-1,-2)@out.grad)
        out._backward=bw; return out
    def sum(self,axis=None,keepdims=False):
        out=Tensor(self.data.sum(axis=axis,keepdims=keepdims),self.requires_grad,(self,))
        def bw():
            g=out.grad
            if axis is not None and not keepdims:
                axes=(axis,) if isinstance(axis,int) else tuple(axis)
                for a in sorted([x if x>=0 else x+self.data.ndim for x in axes]):g=np.expand_dims(g,a)
            if self.requires_grad:self._add_grad(np.ones_like(self.data)*g)
        out._backward=bw; return out
    def mean(self,axis=None,keepdims=False): return self.sum(axis,keepdims)/self.data.size if axis is None else self.sum(axis,keepdims)/np.prod(np.asarray(self.data.shape)[list((axis,) if isinstance(axis,int) else axis)])
    def relu(self):
        out=Tensor(np.maximum(self.data,0),self.requires_grad,(self,)); out._backward=lambda:self._add_grad(out.grad*(self.data>0)) if self.requires_grad else None; return out
    def tanh(self):
        out=Tensor(np.tanh(self.data),self.requires_grad,(self,)); out._backward=lambda:self._add_grad(out.grad*(1-out.data*out.data)) if self.requires_grad else None; return out
    def _add_grad(self,g): self.grad=g if self.grad is None else self.grad+g
    def backward(self,grad=None):
        topo=[]; seen=set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for x in v._prev:build(x)
                topo.append(v)
        build(self)
        if not self._prev:
            seed=np.ones_like(self.data) if grad is None else np.asarray(grad,dtype=np.float64)
            if seed.shape!=self.data.shape:raise ValueError('seed shape mismatch')
            self._add_grad(seed)
            return
        for v in topo:
            if v._prev: v.grad=None
        self.grad=np.ones_like(self.data) if grad is None else np.asarray(grad,dtype=np.float64)
        if self.grad.shape != self.data.shape: raise ValueError('seed shape mismatch')
        for v in reversed(topo): v._backward()

def cross_entropy(logits, labels):
    z=logits.data; y=np.asarray(labels)
    if z.ndim!=2 or len(z)==0 or y.shape!=(len(z),) or not np.issubdtype(y.dtype,np.integer) or np.any(y<0) or np.any(y>=z.shape[1]): raise ValueError('invalid logits or labels')
    shifted=z-z.max(1,keepdims=True); e=np.exp(shifted); p=e/e.sum(1,keepdims=True); loss=(np.log(e.sum(1))-shifted[np.arange(len(y)),y]).mean(); out=Tensor(loss,logits.requires_grad,(logits,))
    def bw():
        if logits.requires_grad:
            g=p.copy();g[np.arange(len(y)),y]-=1;logits._add_grad(g/len(y)*out.grad)
    out._backward=bw;return out
