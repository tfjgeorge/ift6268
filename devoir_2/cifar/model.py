from layers import convolutional_bias, activation, maxpool, batch_norm, avgpool
import theano
from theano import tensor

def multi_dim_softmax(X):
	"""
	compute a softmax for a filter_map at each point

	X : a 4d tensor (batch, feature_map, x, y)
	returns a 4d tensor (batch, softmax, x, y)
	"""
	maxs = X.max(axis=1, keepdims=True)
	maxs = tensor.addbroadcast(maxs, 1)
	exps = tensor.exp(X-maxs)
	sums = exps.sum(axis=1, keepdims=True)
	sums = tensor.addbroadcast(sums, 1)
	return exps/sums


def get_model(X, batch_size, image_dimension):

    input_shape = (batch_size, 3, image_dimension[0], image_dimension[1])
    all_parameters = []
    acc_parameters = []

    #############################################
    # a first convolution with 64 (3, 3) filters
    output, output_test, params, output_shape = convolutional_bias(X, X, input_shape, 64, (3, 3))
    all_parameters += params

    # maxpool with size=(2, 2)
    output, output_test, params, output_shape = maxpool(output, output_test, output_shape, (2, 2))

    # relu activation
    output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')

    #############################################
    # a second convolution with 128 (3, 3) filters
    output, output_test, params, output_shape = convolutional_bias(output, output_test, output_shape, 128, (3, 3))
    all_parameters += params

    # maxpool with size=(2, 2)
    output, output_test, params, output_shape = maxpool(output, output_test, output_shape, (2, 2))

    # relu activation
    output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')

    #############################################
    # 2 convolutional_bias layers with 256 (3, 3) filters
    output, output_test, params, output_shape = convolutional_bias(output, output_test, output_shape, 256, (3, 3))
    all_parameters += params
    output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')
    output, output_test, params, output_shape = convolutional_bias(output, output_test, output_shape, 256, (3, 3))
    all_parameters += params

    # maxpool with size=(2, 2)
    output, output_test, params, output_shape = maxpool(output, output_test, output_shape, (2, 2))

    # relu activation
    output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')

    #############################################
    # Fully connected
    output, output_test, params, output_shape = convolutional_bias(output, output_test, output_shape, 1024, (1, 1))
    all_parameters += params
    output, output_test, params, output_shape = activation(output, output_test, output_shape, 'relu')
    output, output_test, params, output_shape = convolutional_bias(output, output_test, output_shape, 1024, (1, 1))
    all_parameters += params

    # maxpool with size=(4, 4) and fully connected
    output, output_test, params, output_shape = convolutional_bias(output, output_test, output_shape, 110, (1, 1))
    all_parameters += params

    output, output_test, params, output_shape, cacc_parameters = batch_norm(output, output_test, output_shape)
    acc_parameters += cacc_parameters
    
    # softmax
    output_10 = multi_dim_softmax(output[:,:10])
    output_test_10 = multi_dim_softmax(output_test[:,:10])
    
    output_100 = multi_dim_softmax(output[:,10:])
    output_test_100 = multi_dim_softmax(output_test[:,10:])

    #
    return output_10, output_test_10, output_100, output_test_100, all_parameters, acc_parameters
