import numpy as np
import pytest
from nn import MLP,Adam
from prune import prune
from prune.export import SparseMLP
from train.run import train

@pytest.mark.parametrize('target',[0,.5,.9,1])
def test_exact_budget_and_sparse_parity(target):
    model=MLP(np.random.default_rng(0),h=4);opt=Adam(model.params())
    prune(model,target,False,optimizer=opt)
    total=sum(l.mask.size for l in model.layers)
    assert sum(np.count_nonzero(~l.mask) for l in model.layers)==int(np.ceil(target*total))
    for n in [1,8]:
        x=np.random.default_rng(3).normal(size=(n,2))
        assert np.allclose(model.dense_forward(x),SparseMLP(model)(x))

def test_no_implicit_regrowth_and_ties():
    a=MLP(np.random.default_rng(0),h=4);b=MLP(np.random.default_rng(1),h=4)
    for model in [a,b]:
        for l in model.layers:l.w.data[:]=1
        prune(model,.5,False);prune(model,.2,False)
    assert all(np.array_equal(x.mask,y.mask) for x,y in zip(a.layers,b.layers))
    assert sum(l.mask.sum() for l in a.layers)==sum(l.mask.size for l in a.layers)//2

def test_training_finishes_at_budget_without_resurrecting():
    model,history=train(0,20,.9)
    total=sum(l.mask.size for l in model.layers)
    assert sum(np.count_nonzero(~l.mask) for l in model.layers)==int(np.ceil(.9*total))
    assert all(np.all(l.w.data[~l.mask]==0) for l in model.layers)
    assert all(np.isfinite(row[0]) for row in history)

def test_dense_training_learns():
    _,history=train(0,30)
    assert history[-1][0]<history[0][0] and history[-1][1]>.7
