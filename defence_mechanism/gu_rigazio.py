from math import ceil

import numpy as np

from .base import DefenseMechanism


class GuRigazio(DefenseMechanism):
    def __init__(
            self,
            keras_model,
            noise_stddev,
            how,
            *args,  # pylint: disable=W0613
            interpretation=None,
            **kwargs):
        self._model = keras_model
        self._sigma = noise_stddev
        self._how = how
        self._interpretation = interpretation

    def predict_n(self, *batches):
        if self._how == 'L1' and self._interpretation is None:
            yield self._predictions('L1', batches)
        elif self._how in ['L*', 'L+'] and self._interpretation == 'weights':
            orig_weights = self._inject_noise_into_weights()
            yield self._predictions(self._how, batches)
            self._restore_weights(orig_weights)

        elif self._how in ['Lz'] and self._interpretation == 'weights':
            orig_weights = self._inject_noise_into_random_positions()
            yield self._predictions(self._how, batches)
            self._restore_weights(orig_weights)
        else:
            raise ValueError(
                'Not supported combination of `how`: {} '
                'and `interpretation`: {}.'.format(
                    self._how, self._interpretation
                )
            )

    def _predictions(self, how, batches):
        batch_transformer = {
            'L1': self._noisy_layer,
            'L*': self._noisy_layer,
            'L+': lambda x: x,
            'Lz': lambda x: x,
        }[how]
        return tuple(
            self._model.predict(batch_transformer(batch))
            for batch in batches
        )

    def _inject_noise_into_weights(self):
        orig_weights = self._model.get_weights()
        noisy_weights = self._noisy_weights(orig_weights)
        self._model.set_weights(noisy_weights)
        return orig_weights

    def _noisy_weights(self, weights):
        return [self._noisy_layer(layer) for layer in weights]

    def _inject_noise_into_random_positions(self):
        orig_weights = self._model.get_weights()
        noisy_weights = self._noisy_random(orig_weights)
        self._model.set_weights(noisy_weights)
        return orig_weights

    def _noisy_random(self, weights):
        X_noise = np.copy(weights)

        k = np.random.choice(X_noise.shape[0], size=int(
            X_noise.shape[0]/2), replace=False)
        for i in range(k):
            self._noisy_layer(X_noise[i])
        return X_noise

    def _noisy_layer(self, layer):
        expected_value = layer
        np.random.seed()
        return self._sigma * np.random.randn(*layer.shape) + expected_value

    def _restore_weights(self, weights):
        self._model.set_weights(weights)

    @staticmethod
    def parameter_names():
        return ['noise_stddev', 'how', 'interpretation']
