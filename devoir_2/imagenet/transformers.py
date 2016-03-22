from fuel.transformers import SourcewiseTransformer
import numpy
import random

class OneHotEncode10(SourcewiseTransformer):
    """

    """
    def __init__(self, data_stream, **kwargs):
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        super(OneHotEncode10, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        return self._example_transform(example, source_name)

    def _example_transform(self, example, source_name):
        onehot = numpy.zeros((10,))
        onehot[example] = 1

        return onehot

class OneHotEncode100(SourcewiseTransformer):
    """

    """
    def __init__(self, data_stream, **kwargs):
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        super(OneHotEncode100, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        return self._example_transform(example, source_name)

    def _example_transform(self, example, source_name):
        onehot = numpy.zeros((100,))
        onehot[example] = 1

        return onehot


class RandomHorizontalFlip(SourcewiseTransformer):
    """

    """
    def __init__(self, data_stream, **kwargs):
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        super(RandomHorizontalFlip, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        return self._example_transform(example, source_name)

    def _example_transform(self, example, source_name):
        flip = random.randint(0, 1)*2-1
        return example[:,:,::flip]