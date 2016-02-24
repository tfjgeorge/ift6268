from blocks_extras.extensions.plot import Plot

from fuel.datasets import CIFAR10
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from model import get_model
import numpy
import socket
import theano
from theano import tensor
theano.config.floatX = 'float32'
from theano.tensor.nnet import categorical_crossentropy
from transformers import OneHotEncode, RandomHorizontalFlip

batch_size = 32
test_dataset = CIFAR10(('test',), subset=slice(0, 9984))

test_stream = DataStream.default_stream(
    test_dataset,
    iteration_scheme=SequentialScheme(test_dataset.num_examples, batch_size)
)
test_stream = OneHotEncode(test_stream, which_sources=('targets',))

X = tensor.ftensor4('features')
targets = tensor.fmatrix('targets')

output, output_test, all_parameters, acc_parameters = get_model(X, batch_size, (32, 32))

saved_parameters = numpy.load('./best_weights.npz')['arr_0']
for param_w, param in zip(saved_parameters, all_parameters):
    param.set_value(param_w)
    
saved_parameters_bn = numpy.load('./best_weights_bn.npz')['arr_0']
for param_w, param in zip(saved_parameters_bn, acc_parameters):
    param[1].set_value(param_w)

error_test = tensor.neq(tensor.argmax(output_test[:,:,0,0], axis=1), tensor.argmax(targets, axis=1)).mean()
error_test.name = 'error_test'

f_valid = theano.function(
	inputs=[X, targets],
	outputs=[error_test],
)

ei_valid = test_stream.get_epoch_iterator()
valid_loss, valid_error = 0, 0

for batch_num in range(test_dataset.num_examples/batch_size):
	try:
		batch = next(ei_valid)
	except StopIteration:
		continue

	outputs = f_valid(batch[0], batch[1])
	valid_loss += outputs[0]
	valid_error += outputs[1]

valid_loss /= test_dataset.num_examples/batch_size
valid_error /= test_dataset.num_examples/batch_size
print valid_loss, valid_error