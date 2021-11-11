from abc import ABC, abstractmethod

import numpy as np


class Network(ABC):

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

    def labels(self, x):
        predictions = self.predict(x)
        labels = np.argmax(predictions, axis=1)
        return labels

    @staticmethod
    @abstractmethod
    def bounds():
        bgr_mean_pixel = [103.939, 116.779, 123.68]
        bnds = (np.subtract(0, max(bgr_mean_pixel), dtype=np.float32),
                np.subtract(255, min(bgr_mean_pixel), dtype=np.float32))
        return bnds

    @abstractmethod
    def name():
        """
        Returns
        -------
        name : str
            A class-unique human readable identifier of the network.
        """


class KerasNetwork(Network):  # pylint: disable=W0223

    def __init__(self, model):
        self._wrapped_model = model

    def predict(self, x):
        return self._wrapped_model.predict(x)

    def wrapped_model(self):
        return self._wrapped_model
