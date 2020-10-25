from typing import List

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import numpy as np


class Layer(ABC):

    @property
    def weights(self) -> Optional[Tuple[np.array, np.array]]:
        """
        Returns weights tensor if layer is trainable.
        Returns None for non-trainable layers.
        """
        return None

    @property
    def gradients(self) -> Optional[Tuple[np.array, np.array]]:
        """
        Returns bias tensor if layer is trainable.
        Returns None for non-trainable layers.
        """
        return None

    @abstractmethod
    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        """
        Perform layer forward propagation logic.
        """
        pass

    @abstractmethod
    def backward_pass(self, da_curr: np.array) -> np.array:
        pass

    def set_wights(self, w: np.array, b: np.array) -> None:
        """
        Perform layer backward propagation logic.
        """
        pass


class Optimizer(ABC):

    @abstractmethod
    def update(self, layers: List[Layer]) -> None:
        """
        Updates value of weights and bias tensors in trainable layers.
        """
        pass


class GradientDescent(Optimizer):
    def __init__(self, lr: float):
        """
        :param lr - learning rate
        """
        self._lr = lr

    def update(self, layers: List[Layer]) -> None:
        for layer in layers:
            weights, gradients = layer.weights, layer.gradients
            if weights is None or gradients is None:
                continue

            (w, b), (dw, db) = weights, gradients
            layer.set_wights(
                w = w - self._lr * dw,
                b = b - self._lr * db
            )
