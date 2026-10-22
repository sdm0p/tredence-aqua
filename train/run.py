import numpy as np
from nn import MLP,Adam
from engine import Tensor,cross_entropy
def spiral(n=900,seed=0):
 r=np.random.default_rng(seed);X=[];y=[]
 for c in range(3):
  t=np.linspace(c*4,(c+1)*4,n//3)+r.normal(0,.2,n//3); rad=np.linspace(.2,1,n//3);X.extend(np.c_[rad*np.sin(t),rad*np.cos(t)]);y.extend([c]*(n//3))
 return np.asarray(X),np.asarray(y)
def train(seed=0,epochs=80,sparsity=0,method='saliency'):
 X,y=spiral(seed=7); rng=np.random.default_rng(seed);m=MLP(rng);o=Adam(m.params());order=np.arange(len(y));hist=[]
 ema=[np.zeros_like(w.data) for w in m.weights()]
 for ep in range(epochs):
  rng.shuffle(order)
  epoch_loss=0
  for ix in np.array_split(order,10):
   loss=cross_entropy(m(X[ix]),y[ix]);loss.backward()
   epoch_loss+=float(loss.data)*len(ix)
   for j,w in enumerate(m.weights()):ema[j]=.9*ema[j]+.1*np.abs(w.data*w.grad)
   o.step()
  progress=np.clip(((ep+1)-epochs*.2)/(epochs*.5),0,1)
  target=sparsity*(1-(1-progress)**3)
  __import__('prune').prune(m,target,method=='saliency',ema,optimizer=o)
  pred=m.predict(X);hist.append((epoch_loss/len(y),float((pred==y).mean())))
 return m,hist
if __name__=='__main__':
 m,h=train();print(h[-1])
