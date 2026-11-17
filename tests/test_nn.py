# backward passes checked against finite differences in float64

import numpy as np
import pytest

from glassbox.nn import (Conv2D, LeakyReLU, MaxPool2D, ReLU, Sequential,
                         numerical_gradient, relative_error)
from glassbox.nn.layers import col2im, im2col

F64 = np.float64


# definition of a same-padded, stride-1 convolution as four loops
def conv_forward_naive(A_prev, W, b):
    m, n_H, n_W, _ = A_prev.shape
    f, _, _, n_C = W.shape
    p = (f - 1) // 2
    A_prev_pad = np.pad(A_prev, ((0, 0), (p, p), (p, p), (0, 0)))
    Z = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    a_slice = A_prev_pad[i, h:h + f, w:w + f, :]
                    Z[i, h, w, c] = np.sum(a_slice * W[:, :, :, c]) + b[c]
    return Z


# gradient check under J = sum(forward(x) * R), R random so every output matters
def check_layer(layer, x, tolerance=1e-6):
    rng = np.random.default_rng(0)
    R = rng.standard_normal(layer.forward(x).shape)
    J = lambda: float(np.sum(layer.forward(x) * R))

    layer.forward(x)
    dx = layer.backward(R)
    _, dx_approx = numerical_gradient(J, x)
    assert relative_error(dx.ravel(), dx_approx) < tolerance

    for name, p in layer.parameters().items():
        layer.forward(x)
        layer.backward(R)
        grad = p.grad.copy()
        _, grad_approx = numerical_gradient(J, p.data)
        assert relative_error(grad.ravel(), grad_approx) < tolerance, name


@pytest.mark.parametrize("f", [1, 3, 5])
def test_conv_forward_matches_naive_loops(f):
    rng = np.random.default_rng(f)
    conv = Conv2D(3, 4, f, rng, dtype=F64)
    conv.b.data[:] = rng.standard_normal(4)
    A_prev = rng.standard_normal((2, 6, 7, 3))
    np.testing.assert_allclose(conv.forward(A_prev),
                               conv_forward_naive(A_prev, conv.W.data, conv.b.data), atol=1e-12)


# <im2col(x), y> == <x, col2im(y)>
def test_col2im_is_the_adjoint_of_im2col():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 7, 8, 3))
    y = rng.standard_normal((2 * 5 * 6, 3 * 3 * 3))
    lhs = np.sum(im2col(x, 3, 5, 6) * y)
    rhs = np.sum(x * col2im(y, x.shape, 3, 5, 6))
    assert lhs == pytest.approx(rhs)


@pytest.mark.parametrize("f", [1, 3])
def test_conv_gradients(f):
    rng = np.random.default_rng(1)
    check_layer(Conv2D(2, 3, f, rng, dtype=F64), rng.standard_normal((2, 4, 5, 2)))


def test_conv_without_input_grad_still_fills_parameter_grads():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 4, 4, 2))
    full = Conv2D(2, 3, 3, np.random.default_rng(5), dtype=F64)
    lean = Conv2D(2, 3, 3, np.random.default_rng(5), dtype=F64, input_grad=False)
    dZ = rng.standard_normal((2, 4, 4, 3))
    full.forward(x), full.backward(dZ)
    lean.forward(x)
    assert lean.backward(dZ) is None
    np.testing.assert_array_equal(full.W.grad, lean.W.grad)
    np.testing.assert_array_equal(full.b.grad, lean.b.grad)


def test_maxpool_forward():
    x = np.arange(16, dtype=F64).reshape(1, 4, 4, 1)
    np.testing.assert_array_equal(MaxPool2D(2).forward(x)[0, :, :, 0], [[5, 7], [13, 15]])


def test_maxpool_gradient():
    check_layer(MaxPool2D(2), np.random.default_rng(3).standard_normal((2, 4, 6, 3)))


def test_maxpool_shares_gradient_between_ties():
    pool = MaxPool2D(2)
    pool.forward(np.ones((1, 2, 2, 1)))
    np.testing.assert_array_equal(pool.backward(np.ones((1, 1, 1, 1)))[0, :, :, 0],
                                  np.full((2, 2), 0.25))


def test_leaky_relu_gradient():
    x = np.random.default_rng(4).standard_normal((2, 3, 3, 2))
    x += np.sign(x) * 0.1 # stay away from the kink at 0
    check_layer(LeakyReLU(0.1), x)
    np.testing.assert_array_equal(ReLU().forward(np.array([-2.0, 3.0])), [0.0, 3.0])


def test_whole_network_gradient():
    rng = np.random.default_rng(6)
    net = Sequential([
        Conv2D(3, 4, 3, rng, dtype=F64), LeakyReLU(0.1), MaxPool2D(2),
        Conv2D(4, 5, 3, rng, dtype=F64), LeakyReLU(0.1), MaxPool2D(2),
        Conv2D(5, 6, 1, rng, dtype=F64),
    ])
    x = rng.standard_normal((2, 8, 8, 3))
    R = rng.standard_normal((2, 2, 2, 6))
    J = lambda: float(np.sum(net.forward(x) * R))
    for name, p in net.parameters().items():
        net.forward(x)
        net.backward(R)
        grad = p.grad.copy()
        _, grad_approx = numerical_gradient(J, p.data)
        assert relative_error(grad.ravel(), grad_approx) < 1e-6, name


def test_state_dict_round_trip():
    make = lambda seed: Sequential([Conv2D(3, 4, 3, np.random.default_rng(seed)), ReLU(),
                                    Conv2D(4, 2, 1, np.random.default_rng(seed + 1))])
    a, b = make(0), make(10)
    x = np.random.default_rng(0).standard_normal((1, 4, 4, 3)).astype(np.float32)
    assert not np.allclose(a.forward(x), b.forward(x))
    b.load_state_dict({k: v.copy() for k, v in a.state_dict().items()})
    np.testing.assert_array_equal(a.forward(x), b.forward(x))
    assert a.num_parameters() == 3 * 3 * 3 * 4 + 4 + 4 * 2 + 2
    with pytest.raises(KeyError):
        b.load_state_dict({"0.W": a.state_dict()["0.W"]})
