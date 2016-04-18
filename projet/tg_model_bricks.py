import theano
from theano import tensor
from blocks.bricks import BatchNormalization, Rectifier
from blocks.bricks.conv import Convolutional, ConvolutionalSequence
from blocks.initialization import Constant, Uniform
from blocks.bricks.base import application
from blocks.bricks.interfaces import Activation
import numpy

alpha = 0.01
MEAN_VALUES = numpy.array([104, 117, 123]).reshape((1,3,1,1))

class LeakyReLU(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.switch(input_ > 0, input_, alpha * input_)

class BoundedReLU(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.switch(input_ > -1,
            tensor.switch(input_ > 1, alpha * input_ + (1-alpha), input_),
            alpha * input_ - (1-alpha))

def join(small_image, big_image, n_filter_small, n_filter_big, big_img_size_in, ordering=''):

    # upsample small image
    upsampled_small = tensor.repeat(small_image, 2, axis=2)
    upsampled_small = tensor.repeat(upsampled_small, 2, axis=3)

    img_size_small = (n_filter_small, big_img_size_in[0], big_img_size_in[1])
    img_size_big = (n_filter_big, big_img_size_in[0], big_img_size_in[1])

    bn_small = BatchNormalization(img_size_small, name='bn_small%s' % (ordering,))
    bn_small.initialize()
    bn_big = BatchNormalization(img_size_big, name='bn_big%s' % (ordering,))
    bn_big.initialize()

    depth_concat = tensor.concatenate([bn_small.apply(upsampled_small), bn_big.apply(big_image)], axis=1)

    return depth_concat

def conv_block(input_img, n_filter, filter_size, input_featuremap_size, ordering=''):

    # found in torch spatialconvolution
    std0 = 2. / (filter_size[0]*filter_size[1]*input_featuremap_size[0]) ** .5
    std1 = 2. / (input_featuremap_size[0]) ** .5

    layers = []
    layers.append(Convolutional(filter_size=filter_size,
        num_filters=n_filter, border_mode='half',
        name='conv%s_1' % (ordering,), use_bias=True,
        weights_init=Uniform(width=std0)))
    layers.append(BatchNormalization(name='bn%s_1' % (ordering,)))
    layers.append(LeakyReLU())
    layers.append(Convolutional(filter_size=filter_size,
        num_filters=n_filter, border_mode='half',
        name='conv%s_2' % (ordering,), use_bias=True,
        weights_init=Uniform(width=std0)))
    layers.append(BatchNormalization(name='bn%s_2' % (ordering,)))
    layers.append(LeakyReLU())
    layers.append(Convolutional(filter_size=(1, 1),
        num_filters=n_filter, border_mode='valid',
        name='conv%s_3b' % (ordering,), use_bias=True,
        weights_init=Uniform(width=std1)))
    layers.append(BatchNormalization(name='bn%s_3' % (ordering,)))
    layers.append(LeakyReLU())

    conv_sequence = ConvolutionalSequence(layers,
        num_channels=input_featuremap_size[0],
        image_size=(input_featuremap_size[1], input_featuremap_size[2]),
        biases_init=Uniform(width=.1), name='convsequence%s' % (ordering,))

    conv_sequence.initialize()
    return conv_sequence.apply(input_img)

def build_model(Zs):

    # first block
    z1 = conv_block(Zs['Z1'], 8, (3, 3), (3, 256, 256), '1')
    z2 = conv_block(Zs['Z2'], 8, (3, 3), (3, 128, 128), '2')
    z3 = conv_block(Zs['Z3'], 8, (3, 3), (3, 64, 64), '3')
    z4 = conv_block(Zs['Z4'], 8, (3, 3), (3, 32, 32), '4')
    z5 = conv_block(Zs['Z5'], 8, (3, 3), (3, 16, 16), '5')

    # merge 4 & 5
    m45 = join(z5, z4, 8, 8, (32, 32), '45')

    # merge 3 & 45
    c45 = conv_block(m45, 16, (3, 3), (16, 32, 32), '45')
    m345 = join(c45, z3, 16, 8, (64, 64))

    # merge 2 & 345
    c345 = conv_block(m345, 24, (3, 3), (24, 64, 64), '345')
    m2345 = join(c345, z2, 24, 8, (128, 128), '345')

    # merge 1 & 2345
    c2345 = conv_block(m2345, 32, (3, 3), (32, 128, 128), '2345')
    m12345 = join(c2345, z1, 32, 8, (256, 256), '2345')

    last_conv_block = conv_block(m12345, 40, (3, 3), (40, 256, 256), '12345')

    conv_out = Convolutional((1, 1), 3, 40, image_size=(256, 256),
        biases_init=Uniform(width=.1), weights_init=Uniform(width=.3),
        use_bias=True, name='conv_out')
    conv_out.initialize()
    conv_out_output = conv_out.apply(last_conv_block)

    mean = tensor.addbroadcast(theano.shared(MEAN_VALUES).astype('float32'), 0, 2, 3)

    return 128. * (1 + conv_out_output) - mean
