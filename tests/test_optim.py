# first steps of each optimizer, computed by hand from its update rule

import numpy as np
import pytest

from glassbox.nn import Parameter
from glassbox.optim import SGD, Adam, Momentum, RMSprop


def make_param(value, grad):
    p = Parameter(np.array(value, dtype=np.float64), kind="weight")
    p.grad[...] = grad
    return p


def test_sgd_step():
    p = make_param([1.0, -2.0], [0.5, 4.0])
    SGD([p], lr=0.1).step()
    np.testing.assert_allclose(p.data, [0.95, -2.4])


def test_momentum_two_steps():
    p = make_param([1.0], [2.0])
    opt = Momentum([p], lr=0.1, beta=0.9)
    opt.step() # v = 0.1 * 2 = 0.2
    np.testing.assert_allclose(p.data, [1.0 - 0.1 * 0.2])
    p.grad[...] = [4.0]
    opt.step() # v = 0.9 * 0.2 + 0.1 * 4 = 0.58
    np.testing.assert_allclose(p.data, [1.0 - 0.1 * 0.2 - 0.1 * 0.58])


def test_rmsprop_step():
    p = make_param([1.0], [2.0])
    RMSprop([p], lr=0.1, beta=0.9, epsilon=1e-8).step() # s = 0.1 * 4 = 0.4
    np.testing.assert_allclose(p.data, [1.0 - 0.1 * 2.0 / (np.sqrt(0.4) + 1e-8)])


def test_adam_two_steps():
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.01
    p = make_param([1.0], [2.0])
    opt = Adam([p], lr=lr, beta1=b1, beta2=b2, epsilon=eps)

    opt.step()
    v, s = (1 - b1) * 2.0, (1 - b2) * 4.0
    w = 1.0 - lr * (v / (1 - b1)) / (np.sqrt(s / (1 - b2)) + eps)
    np.testing.assert_allclose(p.data, [w])
    assert abs(1.0 - p.data[0]) == pytest.approx(lr, rel=1e-6) # first step has size lr

    p.grad[...] = [-1.0]
    opt.step()
    v, s = b1 * v + (1 - b1) * -1.0, b2 * s + (1 - b2) * 1.0
    w -= lr * (v / (1 - b1 ** 2)) / (np.sqrt(s / (1 - b2 ** 2)) + eps)
    np.testing.assert_allclose(p.data, [w])


@pytest.mark.parametrize("make", [
    lambda p: SGD(p, lr=0.02), lambda p: Momentum(p, lr=0.02),
    lambda p: RMSprop(p, lr=0.01, beta=0.9), lambda p: Adam(p, lr=0.05),
])
def test_optimizers_minimize_an_elongated_bowl(make):
    p = make_param([3.0, 2.0], [0.0, 0.0])
    opt = make([p])
    for _ in range(1500):
        p.grad[...] = p.data * [1.0, 25.0] # J = 0.5 * (w0^2 + 25 * w1^2)
        opt.step()
    assert np.abs(p.data).max() < 0.05


def test_updates_keep_float32():
    p = Parameter(np.ones(3, dtype=np.float32), kind="weight")
    p.grad[...] = 0.5
    for opt in (SGD([p], 0.1), Momentum([p], 0.1), RMSprop([p], 0.1), Adam([p], 0.1)):
        opt.step()
        assert p.data.dtype == np.float32
