# sgd, momentum, rmsprop and adam; each updates param.data in place from param.grad

from __future__ import annotations

import numpy as np

from glassbox.nn.module import Parameter


class Optimizer:
    def __init__(self, params: list[Parameter], lr: float):
        self.params = list(params)
        self.lr = lr
        self.t = 0 # updates taken so far

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            self.update(i, p)

    def update(self, i: int, p: Parameter) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def update(self, i: int, p: Parameter) -> None:
        p.data -= self.lr * p.grad


# exponentially weighted average of past gradients
class Momentum(Optimizer):
    def __init__(self, params: list[Parameter], lr: float, beta: float = 0.9):
        super().__init__(params, lr)
        self.beta = beta
        self.v = [np.zeros_like(p.data) for p in self.params]

    def update(self, i: int, p: Parameter) -> None:
        self.v[i] = self.beta * self.v[i] + (1 - self.beta) * p.grad
        p.data -= self.lr * self.v[i]


# scale each step by the recent size of its own gradient
class RMSprop(Optimizer):
    def __init__(self, params: list[Parameter], lr: float, beta: float = 0.999,
                 epsilon: float = 1e-8):
        super().__init__(params, lr)
        self.beta = beta
        self.epsilon = epsilon
        self.s = [np.zeros_like(p.data) for p in self.params]

    def update(self, i: int, p: Parameter) -> None:
        self.s[i] = self.beta * self.s[i] + (1 - self.beta) * p.grad ** 2
        p.data -= self.lr * p.grad / (np.sqrt(self.s[i]) + self.epsilon)


# momentum and rmsprop together, with bias correction
class Adam(Optimizer):
    def __init__(self, params: list[Parameter], lr: float, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.s = [np.zeros_like(p.data) for p in self.params]

    def update(self, i: int, p: Parameter) -> None:
        self.v[i] = self.beta1 * self.v[i] + (1 - self.beta1) * p.grad
        self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2) * p.grad ** 2
        v_corrected = self.v[i] / (1 - self.beta1 ** self.t)
        s_corrected = self.s[i] / (1 - self.beta2 ** self.t)
        p.data -= self.lr * v_corrected / (np.sqrt(s_corrected) + self.epsilon)
