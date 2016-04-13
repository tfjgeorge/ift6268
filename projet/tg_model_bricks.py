import theano
from theano import tensor
from blocks.bricks import BatchNormalization, Rectifier
from blocks.bricks.conv import Convolutional, ConvolutionalSequence
from blocks.initialization import Constant, Uniform

def join(small_image, big_image, n_filter_small, n_filter_big, big_img_size_in):

    # upsample small image
    upsampled_small = tensor.repeat(small_image, 2, axis=2)
    upsampled_small = tensor.repeat(upsampled_small, 2, axis=3)
    depth_concat = tensor.concatenate([upsampled_small, big_image], axis=1)

    img_size = (n_filter_small + n_filter_big, big_img_size_in[0], big_img_size_in[1])

    bn = BatchNormalization(img_size)
    bn.initialize()

    return bn.apply(depth_concat)

def conv_block(input_img, n_filter, filter_size, input_featuremap_size):

    layers = []
    layers.append(Convolutional(filter_size=filter_size,
        num_filters=n_filter, border_mode='half'))
    layers.append(BatchNormalization())
    layers.append(Rectifier())
    layers.append(Convolutional(filter_size=filter_size,
        num_filters=n_filter, border_mode='half'))
    layers.append(BatchNormalization())
    layers.append(Rectifier())
    layers.append(Convolutional(filter_size=(1, 1),
        num_filters=n_filter, border_mode='valid'))
    layers.append(BatchNormalization())
    layers.append(Rectifier())

    conv_sequence = ConvolutionalSequence(layers,
        num_channels=input_featuremap_size[0],
        image_size=(input_featuremap_size[1], input_featuremap_size[2]),
        biases_init=Constant(0), weights_init=Uniform(width=.1))

    conv_sequence.initialize()
    return conv_sequence.apply(input_img)

def build_model(Zs):

    # first block
    z1 = conv_block(Zs['Z1'], 8, (3, 3), (1, 256, 256))
    z2 = conv_block(Zs['Z2'], 8, (3, 3), (1, 128, 128))
    z3 = conv_block(Zs['Z3'], 8, (3, 3), (1, 64, 64))
    z4 = conv_block(Zs['Z4'], 8, (3, 3), (1, 32, 32))
    z5 = conv_block(Zs['Z5'], 8, (3, 3), (1, 16, 16))

    # merge 4 & 5
    m45 = join(z5, z4, 8, 8, (32, 32))

    # merge 3 & 45
    c45 = conv_block(m45, 16, (3, 3), (16, 32, 32))
    m345 = join(c45, z3, 16, 8, (64, 64))

    # merge 2 & 345
    c345 = conv_block(m345, 24, (3, 3), (24, 64, 64))
    m2345 = join(c345, z2, 24, 8, (128, 128))

    # merge 1 & 2345
    c2345 = conv_block(m2345, 32, (3, 3), (32, 128, 128))
    m12345 = join(c2345, z1, 32, 8, (256, 256))

    last_conv_block = conv_block(m12345, 40, (3, 3), (40, 256, 256))

    conv_out = Convolutional((1, 1), 3, 40, image_size=(256, 256),
        biases_init=Constant(0), weights_init=Uniform(width=.1))
    conv_out.initialize()

    return conv_out.apply(last_conv_block)
