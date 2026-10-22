import numpy as np
import pytest
from engine import Tensor,cross_entropy
from nn import Adam,MLP

def test_mask_lifecycle_preserves_other_coordinate():
    p=Tensor([1.,2.],True);opt=Adam([p]);p.grad=np.ones(2);opt.step()
    opt.set_mask(p,[0,1]);moment=opt.m[0][1]
    for _ in range(3):p.grad=np.ones(2);opt.step()
    assert p.data[0]==opt.m[0][0]==opt.v[0][0]==opt.age[0][0]==0
    age=opt.age[0][1];other_moment=opt.m[0][1]
    opt.set_mask(p,[1,1],revival_value=.5)
    assert opt.age[0][1]==age and opt.m[0][1]==other_moment and other_moment>moment
    fresh=Tensor([.5],True);reference=Adam([fresh]);p.grad=np.ones(2);fresh.grad=np.ones(1)
    opt.step();reference.step();assert p.data[0]==fresh.data[0]

@pytest.mark.parametrize('mask',[[1],[0,2]])
def test_invalid_mask(mask):
    p=Tensor([1.,2.],True)
    with pytest.raises(ValueError):Adam([p]).set_mask(p,mask)

def test_fixed_mask_latent_derivative():
    model=MLP(np.random.default_rng(0),h=4);layer=model.layers[0]
    layer.mask[0,0]=False
    cross_entropy(model([[1.,.4]]),[1]).backward()
    assert layer.w.grad[0,0]==0
    baseline=model.dense_forward([[1.,.4]])
    layer.w.data[0,0]+=100
    assert np.allclose(baseline,model.dense_forward([[1.,.4]]))
