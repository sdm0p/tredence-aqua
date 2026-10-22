"""Independent engine checks; no network or optimizer dependency."""
import numpy as np
import pytest
from engine import Tensor,cross_entropy

@pytest.mark.parametrize('op',['add','subtract','multiply','divide','matmul','sum','mean','relu','tanh'])
def test_vjp_both_operands(op):
    a=np.array([[.4,-.7],[1.2,.5]])
    b=np.array([[1.3,.8],[.9,1.4]])
    def objective(x,y):
        operations={'add':lambda:x+y,'subtract':lambda:x-y,'multiply':lambda:x*y,'divide':lambda:x/y,'matmul':lambda:x@y,'sum':lambda:x.sum(axis=-1),'mean':lambda:x.mean(axis=0),'relu':lambda:x.relu(),'tanh':lambda:x.tanh()}
        return operations[op]().sum()
    x,y=Tensor(a.copy(),True),Tensor(b.copy(),True)
    objective(x,y).backward()
    for data,grad in [(a,x.grad),(b,y.grad)]:
        if grad is None:continue
        for idx in np.ndindex(data.shape):
            old=data[idx];data[idx]=old+1e-6;hi=objective(Tensor(a),Tensor(b)).data
            data[idx]=old-1e-6;lo=objective(Tensor(a),Tensor(b)).data;data[idx]=old
            assert np.allclose(grad[idx],(hi-lo)/2e-6,atol=1e-6)

def test_stable_loss_and_shared_repeated_backward():
    z=Tensor([[1000.,-1000.]],True);loss=cross_entropy(z,[1]);loss.backward()
    assert loss.data==2000 and np.allclose(z.grad,[[1,-1]])
    x=Tensor([2.],True);y=(x*x+x).sum();y.backward();y.backward()
    assert np.allclose(x.grad,[10])

@pytest.mark.parametrize('axis,keepdims',[(None,False),(-1,False),((0,2),False),((0,2),True)])
def test_reduction_axes(axis,keepdims):
    x=Tensor(np.ones((2,3,4)),True);x.mean(axis=axis,keepdims=keepdims).sum().backward()
    denominator=24 if axis is None else (4 if axis==-1 else 8)
    assert np.allclose(x.grad,np.ones((2,3,4))/denominator)

def test_cross_entropy_finite_differences():
    a=np.array([[1.,-.2,.5],[.4,.7,-.9]]);x=Tensor(a.copy(),True);cross_entropy(x,[1,0]).backward()
    for idx in np.ndindex(a.shape):
        old=a[idx];a[idx]=old+1e-6;hi=cross_entropy(Tensor(a),[1,0]).data
        a[idx]=old-1e-6;lo=cross_entropy(Tensor(a),[1,0]).data;a[idx]=old
        assert np.allclose(x.grad[idx],(hi-lo)/2e-6,atol=1e-6)

def test_leaf_root_accumulates_and_seed_shape_is_checked():
    x=Tensor([2.],True);x.backward();x.backward()
    assert np.allclose(x.grad,[2])
    with pytest.raises(ValueError):x.backward([1.,2.])

def test_multiaxis_broadcast_gradient():
    x=Tensor(np.ones((2,3,4)),True);bias=Tensor(np.ones((1,3,1)),True)
    ((x+bias)*2).sum().backward()
    assert np.allclose(bias.grad,np.full((1,3,1),16))
