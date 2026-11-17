# parameter, layer and sequential building blocks (no autograd)

from __future__ import annotations

import numpy as np


# trainable array and its gradient; kind is "weight" or "bias"
class Parameter:
    def __init__(self, data: np.ndarray, kind: str):
        if kind not in ("weight", "bias"):
            raise ValueError(f"kind must be 'weight' or 'bias', got {kind!r}")
        self.data = data
        self.grad = np.zeros_like(data)
        self.kind = kind


# forward caches what backward needs; backward fills param.grad and returns dJ/d(input)
class Layer:
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray | None:
        raise NotImplementedError

    def parameters(self) -> dict[str, Parameter]:
        return {}

    # saved with the model but not trained
    def buffers(self) -> dict[str, np.ndarray]:
        return {}


class Sequential(Layer):
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray | None:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self) -> dict[str, Parameter]:
        return {
            f"{i}.{name}": p
            for i, layer in enumerate(self.layers)
            for name, p in layer.parameters().items()
        }

    def buffers(self) -> dict[str, np.ndarray]:
        return {
            f"{i}.{name}": b
            for i, layer in enumerate(self.layers)
            for name, b in layer.buffers().items()
        }

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {name: p.data for name, p in self.parameters().items()}
        state.update(self.buffers())
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        own = self.state_dict()
        if set(own) != set(state):
            raise KeyError(
                f"state mismatch: missing {sorted(set(own) - set(state))}, "
                f"unexpected {sorted(set(state) - set(own))}"
            )
        for name, array in own.items():
            if array.shape != state[name].shape:
                raise ValueError(
                    f"{name}: shape {state[name].shape} does not fit {array.shape}"
                )
            array[...] = state[name]

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters().values())
