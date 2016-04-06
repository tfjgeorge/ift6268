from lasagne.layers import InputLayer, Upscale2DLayer, DenseLayer, NonlinearityLayer, ConcatLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax

def build_conv(input, name, n_filters):

    net = dict()

    net['conv%s_1' % name] = ConvLayer(
        input, n_filters, 3, pad=1, flip_filters=False)
    net['conv%s_2' % name] = ConvLayer(
        net['conv%s_1' % name], n_filters, 3, pad=1, flip_filters=False)
    net['conv%s_3' % name] = ConvLayer(
        net['conv%s_2' % name], n_filters, 1, pad=0, flip_filters=False)

    return net

def build_model(Zs):
    net = dict()

    net['Z1'] = InputLayer(input_var=Zs['Z1'], shape=(None, 1, 256, 256))
    net['Z2'] = InputLayer(input_var=Zs['Z2'], shape=(None, 1, 128, 128))
    net['Z3'] = InputLayer(input_var=Zs['Z3'], shape=(None, 1, 64, 64))
    net['Z4'] = InputLayer(input_var=Zs['Z4'], shape=(None, 1, 32, 32))
    net['Z5'] = InputLayer(input_var=Zs['Z5'], shape=(None, 1, 16, 16))

    # first block
    net.update(build_conv(net['Z5'], '5', 8))
    net.update(build_conv(net['Z4'], '4', 8))
    net.update(build_conv(net['Z3'], '3', 8))
    net.update(build_conv(net['Z2'], '2', 8))
    net.update(build_conv(net['Z1'], '1', 8))

    # merge 4 & 5
    net['upsample_5'] = Upscale2DLayer(net['conv5_3'], 2)
    net['concat_45'] = ConcatLayer([net['upsample_5'], net['conv4_3']])
    net.update(build_conv(net['concat_45'], '45', 16))

    # merge 3 & 45
    net['upsample_45'] = Upscale2DLayer(net['conv45_3'], 2)
    net['concat_345'] = ConcatLayer([net['upsample_45'], net['conv3_3']])
    net.update(build_conv(net['concat_345'], '345', 24))

    # merge 2 & 345
    net['upsample_345'] = Upscale2DLayer(net['conv345_3'], 2)
    net['concat_2345'] = ConcatLayer([net['upsample_345'], net['conv2_3']])
    net.update(build_conv(net['concat_2345'], '2345', 32))

    # merge 1 & 2345
    net['upsample_2345'] = Upscale2DLayer(net['conv2345_3'], 2)
    net['concat_12345'] = ConcatLayer([net['upsample_2345'], net['conv1_3']])
    net.update(build_conv(net['concat_12345'], '12345', 32))

    net['output'] = ConvLayer(
        net['conv12345_3'], 3, 1, pad=0, flip_filters=False)

    return net