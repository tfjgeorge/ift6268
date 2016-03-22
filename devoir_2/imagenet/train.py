from blocks_extras.extensions.plot import Plot

from blocks.algorithms import GradientDescent, Adam
from blocks.extensions import Printing, Timing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop

from fuel.datasets import CIFAR10
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import ScaleAndShift, Cast

import lasagne

import numpy
import socket
import theano
from theano import tensor
from theano.tensor.nnet import categorical_crossentropy
from transformers import OneHotEncode10, RandomHorizontalFlip, MinimumImageDimensions
import pickle

# max = 45000
n_ex = 40000/32*32
suffix = ''

running_on_laptop = socket.gethostname() == 'yop'
if running_on_laptop:
	from model_cpu import build_model
	host_plot = 'http://localhost:5006'
	slice_train = slice(0, 32*10)
	slice_valid = slice(32*10, 32*10+32*5)
	slice_test = slice(32*15, 32*15+32*5)
else:
	from model import build_model
	host_plot = 'http://hades.calculquebec.ca:5042'
	slice_train = slice(0,n_ex)
	slice_test = slice(45000,50000-8)
	slice_valid = slice(40000,45000-8)


## Load cifar10 stream
batch_size = 32
num_train_example = slice_train.stop - slice_train.start
num_valid_example = slice_valid.stop - slice_valid.start
num_test_example = slice_test.stop - slice_test.start

train_dataset = CIFAR10(('train',), subset=slice_train)
train_stream = DataStream.default_stream(
    train_dataset,
    iteration_scheme=SequentialScheme(train_dataset.num_examples, batch_size)
)
train_stream = OneHotEncode10(train_stream, which_sources=('targets',))
train_stream = RandomHorizontalFlip(train_stream, which_sources=('features',))
train_stream = MinimumImageDimensions(train_stream, (224, 224), which_sources=('features',))
train_stream = ScaleAndShift(train_stream, 1./255, 0, which_sources=('features',))
train_stream = Cast(train_stream, 'floatX', which_sources=('features',))

valid_dataset = CIFAR10(('train',), subset=slice_valid)
valid_stream = DataStream.default_stream(
    valid_dataset,
    iteration_scheme=SequentialScheme(valid_dataset.num_examples, batch_size)
)
valid_stream = OneHotEncode10(valid_stream, which_sources=('targets',))
valid_stream = MinimumImageDimensions(valid_stream, (224, 224), which_sources=('features',))
valid_stream = ScaleAndShift(valid_stream, 1./255, 0, which_sources=('features',))
valid_stream = Cast(valid_stream, 'floatX', which_sources=('features',))


test_dataset = CIFAR10(('train',), subset=slice_test)
test_stream = DataStream.default_stream(
    test_dataset,
    iteration_scheme=SequentialScheme(test_dataset.num_examples, batch_size)
)
test_stream = OneHotEncode10(test_stream, which_sources=('targets',))
test_stream = MinimumImageDimensions(test_stream, (224, 224), which_sources=('features',))
test_stream = ScaleAndShift(test_stream, 1./255, 0, which_sources=('features',))
test_stream = Cast(test_stream, 'floatX', which_sources=('features',))

## build computational graph
X = tensor.ftensor4('features')
targets = tensor.fmatrix('targets')

net = build_model()
# loading pretrained weights
model = pickle.load(open('./blvc_googlenet.pkl'))
lasagne.layers.set_all_param_values(net['prob'], model['param values'])

googlenet_features = lasagne.layers.get_output(net['pool5/7x7_s1'], X)

# add a mlp on top of this
W = theano.shared(
	numpy.random.uniform(low=-0.1, high=0.1, size=(1024, 10)).astype(numpy.float32),
	'linear_weights'
)
b = theano.shared(numpy.zeros(10).astype(numpy.float32))
all_parameters = [W, b]

output = tensor.dot(googlenet_features, W) + b
pred = tensor.nnet.softmax(output)

loss = categorical_crossentropy(pred, targets).mean()
loss.name = 'loss'

loss_test = categorical_crossentropy(pred, targets).mean()
loss.name = 'loss_test'

error = tensor.neq(tensor.argmax(pred, axis=1), tensor.argmax(targets, axis=1)).mean()
error.name = 'error'

error_test = tensor.neq(tensor.argmax(pred, axis=1), tensor.argmax(targets, axis=1)).mean()
error.name = 'error_test'

# construct update rule
learning_rate = 0.01
updates, updates_stats = [], []
for param in all_parameters:
	updates.append((param, param - learning_rate * tensor.grad(loss, param)))

f_train = theano.function(
	inputs=[X, targets],
	outputs=[loss, error],
	updates=updates
)

f_valid = theano.function(
	inputs=[X, targets],
	outputs=[loss_test, error_test],
)

best_valid_error = float('inf')
train_loss_hist, train_error_hist, valid_loss_hist, valid_error_hist, test_loss_hist, test_error_hist = [],[],[],[],[],[]

epoch = 0
patience = 5

# while True:
for i in range(5):
	epoch += 1

	ei_train = train_stream.get_epoch_iterator()
	ei_valid = valid_stream.get_epoch_iterator()
	ei_test = test_stream.get_epoch_iterator()

	train_loss, train_error, valid_loss, valid_error, test_loss, test_error = 0, 0, 0, 0, 0, 0
			
	for batch_num in range(num_train_example/batch_size):
		try:
			batch = next(ei_train)
		except StopIteration:
			continue

		outputs = f_train(batch[0], batch[1])
		train_loss += outputs[0]
		train_error += outputs[1]

	for batch_num in range(num_valid_example/batch_size):
		try:
			batch = next(ei_valid)
		except StopIteration:
			continue

		outputs = f_valid(batch[0], batch[1])
		valid_loss += outputs[0]
		valid_error += outputs[1]

	for batch_num in range(num_test_example/batch_size):
		try:
			batch = next(ei_test)
		except StopIteration:
			continue

		outputs = f_valid(batch[0], batch[1])
		test_loss += outputs[0]
		test_error += outputs[1]

	train_error /= float(num_train_example)/batch_size
	train_loss /= float(num_train_example)/batch_size
	valid_error /= float(num_valid_example)/batch_size
	valid_loss /= float(num_valid_example)/batch_size
	test_error /= float(num_valid_example)/batch_size
	test_loss /= float(num_valid_example)/batch_size

	train_error_hist.append(train_error)
	train_loss_hist.append(train_loss)
	valid_error_hist.append(valid_error)
	valid_loss_hist.append(valid_loss)
	test_error_hist.append(test_error)
	test_loss_hist.append(test_loss)

	print 'epoch %i, train: %.2f, %.2f, valid: %.2f, %.2f, test: %.2f, %.2f' % (epoch, train_loss, train_error, valid_loss, valid_error, test_loss, test_error)

	if valid_error < best_valid_error:
		patience = 5
		best_valid_error = valid_error

		print 'improved valid error'
		# save weights
		numpy.savez('best_weights_%s.npz' % (suffix,), [param.get_value() for param in all_parameters])
		# save batch norm weights
		numpy.savez('best_weights_bn_%s.npz' % (suffix,), [acc.get_value() for val, acc in acc_parameters])
	else:
		patience -= 1
		if patience == 0:
			break

# save train and valid hist
numpy.savez('plots_%s.npz' % (suffix,), [train_loss_hist, train_error_hist, valid_loss_hist, valid_error_hist, test_loss_hist, test_error_hist])
