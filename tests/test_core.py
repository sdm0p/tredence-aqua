import numpy as np
from engine import Tensor,cross_entropy
def test_broadcast():
 x=Tensor(np.ones((3,2)),True);b=Tensor(np.ones(2),True);(x+b).sum().backward();assert np.allclose(b.grad,[3,3])
def test_ce():
 z=Tensor(np.array([[2.,-1.,0.],[-1.,2.,0.]]),True);l=cross_entropy(z,[0,1]);l.backward();assert np.isfinite(l.data) and np.allclose(z.grad.sum(1),0)
