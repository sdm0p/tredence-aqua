import numpy as np
from nn import Adam
from engine import Tensor
def test_fresh_revival_state():
    a=Tensor([1.],True); opt=Adam([a],lr=.1); a.grad=np.array([2.]); opt.step(); opt.set_mask(a,[0]);opt.set_mask(a,[1])
    b=Tensor([0.],True); fresh=Adam([b],lr=.1); a.grad=np.array([1.]);b.grad=np.array([1.]);opt.step();fresh.step();assert np.allclose(a.data,b.data)
