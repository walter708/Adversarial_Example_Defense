import logging
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from sklearn import preprocessing


class DefenseMechanism(ABC):

    def predict(self, batch):
        return (confidences_batch for (confidences_batch,) in self.predict_n(batch))

    def predict_2(self, batch0, batch1):
        return self.predict_n(batch0, batch1)

    @abstractmethod
    def predict_n(self, *batches):
        raise NotImplementedError

    @abstractmethod
    def parameter_names(self):
        return NotImplementedError


def aggregate_predict_n_by(methods, predict_n_result):

        if any(method not in _AVAILABLE_AGGREGATORS.keys() for method in methods):
            raise ValueError(
                'Unsupported aggregation method selection: {}'.format(methods)
            )
        predict_n_result_iter = iter(predict_n_result)

        aggregators = {}
        try:
            # Pulling out the first alternative is necessary to initialize the
            # consumer objects.
            alternative_1_batches = next(predict_n_result_iter)
            for method in methods:
                aggregators[method] = tuple(
                    _AVAILABLE_AGGREGATORS[method](representative=batch)
                    for batch in alternative_1_batches
                )
            _push_into_aggregators(aggregators, alternative_1_batches)
        except StopIteration:
            logging.error(
                '%s %s.%s(): '
                'Provided an empty iterable for argument `predict_n_result`.',
                datetime.now().isoformat(),
                aggregate_predict_n_by.__module__,
                aggregate_predict_n_by.__qualname__,
            )
            raise ValueError(
                'Provided an empty iterable for argument `predict_n_result`.'
            )
        # the remaining alternatives
        for batches in predict_n_result_iter:
            _push_into_aggregators(aggregators, batches)

        ccs = _aggregators_to_aggregation(aggregators)
        return ccs

def _push_into_aggregators(ccs, batches):
        for aggregator_tuple in ccs.values():
            for (aggregator, batch) in zip(aggregator_tuple, batches):
                aggregator.consume(batch)

def _aggregators_to_aggregation(aggregators):
        ccs = {}
        for (method, aggregator_tuple) in aggregators.items():
            ccs[method] = [aggr.aggregation() for aggr in aggregator_tuple]
        return ccs

class CountAggregator():

        def __init__(self, representative, normalize=False):
            self._normalize = normalize
            d_type = np.float_ if normalize else np.int_
            self._counts = np.zeros_like(representative, dtype=d_type)
            self._eye = np.eye(self._counts.shape[1], dtype=d_type)

        def consume(self, array_like):
            assert self._counts.shape == array_like.shape, 'Shape mismatch: {} vs {}'.format(
                self._counts.shape, array_like.shape
            )
            self._counts += self._to_0_1_array(array_like)

        def _to_0_1_array(self, array):
            """
            For every row in the argument array: Set every entry to 0, except the
            maximum value of that row -- set that one to 1.
            """
            indices = np.argmax(array, axis=1)
            return np.squeeze(self._eye[indices])

        def aggregation(self):
            if self._normalize:
                return preprocessing.normalize(self._counts, norm='l1')
            return self._counts

class MeanAggregator():
        """
        Like `CountAggregator` but instead of counting the number of times an
        ensemble member predicts a class for an input, average the vote of each
        ensemble member into one vote.
        """

        def __init__(self, representative, normalize=True):
            self._normalize = normalize
            self._mean = np.zeros_like(representative)
            self._processed = 0

        def consume(self, array_like):
            assert self._mean.shape == array_like.shape, 'Shape mismatch: {} vs {}'.format(
                self._mean.shape, array_like.shape
            )
            self._processed += 1
            self._mean += (array_like - self._mean) / self._processed

        def aggregation(self):
            if self._normalize:
                return preprocessing.normalize(self._mean, norm='l1')
            return self._mean

class TrivialAggregator():
        """
        Just keep the first seen array. This is useful to maintain the interface
        when 'aggregating' singletons.
        """

        def __init__(self, representative):
            self._representative = representative

        def consume(self, *args, **kwargs):
            pass

        def aggregation(self):
            return self._representative

_AVAILABLE_AGGREGATORS = {'count': CountAggregator,
                              'mean': MeanAggregator,
                              'trivial': TrivialAggregator, }

def curried_aggregate_predict_n_by(methods):
        """
        Curried variant of `aggregate_predict_n_by`.
        """
        def _aggregate_predict_n_by(predict_n_result):
            return aggregate_predict_n_by(methods, predict_n_result)
        return _aggregate_predict_n_by
