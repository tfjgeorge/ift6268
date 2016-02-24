import theano
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet import relu
from blocks.graph import add_annotation, Annotation
import numpy

def convolutional(X, X_test, input_shape, n_filters, filter_size):
	"""
	Implementation of a convolutional layer

	Parameters
	----------
	X
	input_shape
	n_filters
	filter_size

	Note
	----
	The convolutions are implemented using border_mode=same, that is the 
	output shape is the same as the input shape for the 2 last dimensions
	"""

	filters_shape = (n_filters, input_shape[1], filter_size[0], filter_size[1])
	filters = theano.shared(
		numpy.random.uniform(low=-0.1, high=0.1, size=filters_shape).astype(numpy.float32),
		'conv_filters'
	)

	output_shape = (input_shape[0], n_filters, input_shape[2], input_shape[3])

	output = conv2d(input=X, filters=filters, filter_shape=filters_shape, image_shape=input_shape, border_mode='full')
	output_test = conv2d(input=X_test, filters=filters, filter_shape=filters_shape, image_shape=input_shape, border_mode='full')

	shift_x = (filter_size[0] - 1) // 2
	shift_y = (filter_size[1] - 1) // 2

	output = output[:,:,shift_x:input_shape[2]+shift_x,shift_y:input_shape[3]+shift_y]
	output_test = output_test[:,:,shift_x:input_shape[2]+shift_x,shift_y:input_shape[3]+shift_y]

	return output, output_test, [filters], output_shape

def maxpool(X, X_test, input_shape, size):
	"""
	A maxpool layer
	"""

	pooled = max_pool_2d(input=X, ds=size, ignore_border=True)
	pooled_test = max_pool_2d(input=X_test, ds=size, ignore_border=True)
	output_shape = (input_shape[0], input_shape[1], input_shape[2]/size[0], input_shape[3]/size[1])

	return pooled, pooled_test, [], output_shape

def avgpool(X, X_test, input_shape, size):
	"""
	A maxpool layer
	"""

	pooled = max_pool_2d(input=X, ds=size, ignore_border=True,  mode='average_exc_pad')
	pooled_test = max_pool_2d(input=X_test, ds=size, ignore_border=True, mode='average_exc_pad')
	output_shape = (input_shape[0], input_shape[1], input_shape[2]/size[0], input_shape[3]/size[1])

	return pooled, pooled_test, [], output_shape



def activation(X, X_test, input_shape, activation_type='relu'):

	if activation_type=='relu':
		output = relu(X)
		output_test = relu(X_test)
	elif activation_type=='sigmoid':
		output = theano.tensor.nnet.sigmoid(X)
		output_test = theano.tensor.nnet.sigmoid(X_test)

	else:
		raise Exception('this non linearity does not exist: %s' % activation_type)

	return output, output_test, [], input_shape

class CustomAnnotation(Annotation):
	pass


def batch_norm(X, X_test, input_shape):

	epsilon = 10**-3

	means = theano.tensor.mean(X, axis=0)
	variances = theano.tensor.var(X, axis=0)
	gammas = theano.shared( numpy.ones((1, input_shape[1], input_shape[2], input_shape[3])) )
	betas = theano.shared( numpy.zeros((1, input_shape[1], input_shape[2], input_shape[3])) )

	means_test = theano.shared( numpy.zeros((1, input_shape[1], input_shape[2], input_shape[3])) )
	variances_test = theano.shared( numpy.zeros((1, input_shape[1], input_shape[2], input_shape[3])) )

	output = theano.tensor.addbroadcast(gammas, 0) * (X - means) / (variances + epsilon)**0.5 + theano.tensor.addbroadcast(betas, 0)
	output_test = theano.tensor.addbroadcast(gammas, 0) * (X - means_test) / (variances_test + epsilon)**0.5 + theano.tensor.addbroadcast(betas, 0)

	return output, X_test, [gammas, betas], input_shape, [(means, means_test), (variances, variances_test)]
