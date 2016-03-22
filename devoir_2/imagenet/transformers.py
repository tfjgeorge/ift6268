from fuel.transformers import SourcewiseTransformer
import numpy
import random
from PIL import Image
import math

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


class MinimumImageDimensions(SourcewiseTransformer):
    """
    """
    def __init__(self, data_stream, shape, resample='nearest',
                 **kwargs):
        self.shape = shape
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        super(MinimumImageDimensions, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        return self._example_transform(example, source_name)

    def _example_transform(self, example, _):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        min_height, min_width = self.shape
        original_height, original_width = example.shape[-2:]
        if original_height < min_height or original_width < min_width:
            dt = example.dtype
            # If we're dealing with a colour image, swap around the axes
            # to be in the format that PIL needs.
            if example.ndim == 3:
                im = example.transpose(1, 2, 0)
            else:
                im = example
            im = Image.fromarray(im)
            width, height = im.size
            multiplier = max(min_width / width, min_height / height)
            width = int(math.ceil(width * multiplier))
            height = int(math.ceil(height * multiplier))
            im = numpy.array(im.resize((width, height), self.resample)).astype(dt)
            # If necessary, undo the axis swap from earlier.
            if im.ndim == 3:
                example = im.transpose(2, 0, 1)
            else:
                example = im
        return example