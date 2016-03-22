from blocks_extras.extensions.plot import Plot

from blocks.algorithms import GradientDescent, Adam
from blocks.extensions import Printing, Timing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop

from fuel.datasets import CIFAR10, CIFAR100
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from model import get_model
import numpy
import socket
import theano
from theano import tensor
from theano.tensor.nnet import categorical_crossentropy
from transformers import OneHotEncode10, OneHotEncode100, RandomHorizontalFlip

# max = 45000
n_ex = 1024
alpha = 0.1
learning_rate = 0.01

suffix = 'alpha%.2f_nex%d' % (alpha, n_ex)

running_on_laptop = socket.gethostname() == 'yop'
if running_on_laptop:
	host_plot = 'http://localhost:5006'
	slice_train = slice(0, 32*10)
	slice_valid = slice(32*10, 32*10+32*5)
	slice_test = slice(32*15, 32*15+32*5)
	slice_100 = slice(0, 32*10)
else:
	host_plot = 'http://hades.calculquebec.ca:5042'
	slice_train = slice(0,n_ex)
	slice_test = slice(45000,50000-8)
	slice_valid = slice(40000,45000-8)
	slice_100 = slice(0, 50000)


## Load cifar10 stream
batch_size = 32
num_train_example = slice_train.stop - slice_train.start
num_valid_example = slice_valid.stop - slice_valid.start
num_test_example = slice_test.stop - slice_test.start
num_train_cifar100 = slice_100.stop - slice_100.start

train_dataset = CIFAR10(('train',), subset=slice_train)
train_stream = DataStream.default_stream(
    train_dataset,
    iteration_scheme=SequentialScheme(train_dataset.num_examples, batch_size)
)
train_stream = OneHotEncode10(train_stream, which_sources=('targets',))
train_stream = RandomHorizontalFlip(train_stream, which_sources=('features',))

valid_dataset = CIFAR10(('train',), subset=slice_valid)
valid_stream = DataStream.default_stream(
    valid_dataset,
    iteration_scheme=SequentialScheme(valid_dataset.num_examples, batch_size)
)
valid_stream = OneHotEncode10(valid_stream, which_sources=('targets',))

test_dataset = CIFAR10(('train',), subset=slice_test)
test_stream = DataStream.default_stream(
    test_dataset,
    iteration_scheme=SequentialScheme(test_dataset.num_examples, batch_size)
)
test_stream = OneHotEncode10(test_stream, which_sources=('targets',))

## Load cifar100 stream
batch_size = 32
num_train_example = slice_train.stop - slice_train.start
num_test_example = slice_test.stop - slice_test.start

train_dataset_100 = CIFAR100(('train',), subset=slice_100)
train_stream_100 = DataStream.default_stream(
    train_dataset_100,
    iteration_scheme=SequentialScheme(train_dataset_100.num_examples, batch_size)
)
train_stream_100 = OneHotEncode100(train_stream_100, which_sources=('fine_labels',))

lr_cifar100 = learning_rate# * num_train_example/num_train_cifar100

## build computational graph
X = tensor.ftensor4('features')
targets = tensor.fmatrix('targets')
targets_100 = tensor.fmatrix('fine_labels')

output_10, output_test_10, output_100, output_test_100, all_parameters, acc_parameters = get_model(X, batch_size, (32, 32))


loss = alpha * categorical_crossentropy(output_10[:,:,0,0], targets).mean()
loss.name = 'loss'

loss_100 = (1-alpha) * categorical_crossentropy(output_100[:,:,0,0], targets_100).mean()
loss_100.name = 'loss_100'

loss_test = categorical_crossentropy(output_test_10[:,:,0,0], targets).mean()
loss.name = 'loss_test'

loss_100_test = categorical_crossentropy(output_test_100[:,:,0,0], targets_100).mean()
loss_100_test.name = 'loss_100_test'

error = tensor.neq(tensor.argmax(output_10[:,:,0,0], axis=1), tensor.argmax(targets, axis=1)).mean()
error.name = 'error'

error_test = tensor.neq(tensor.argmax(output_test_10[:,:,0,0], axis=1), tensor.argmax(targets, axis=1)).mean()
error_test.name = 'error_test'

# construct update rule
updates, updates_100, updates_stats = [], [], []
for param in all_parameters:
	updates.append((param, param - learning_rate * tensor.grad(loss, param)))
	updates_100.append((param, param - lr_cifar100 * tensor.grad(loss_100, param)))
for val, acc in acc_parameters:
	updates_stats.append((acc, acc + val/(num_train_example/batch_size)))

f_train = theano.function(
	inputs=[X, targets],
	outputs=[loss, error],
	updates=updates
)

f_train_100 = theano.function(
	inputs=[X, targets_100],
	outputs=[loss_100],
	updates=updates_100
)

f_stats = theano.function(
	inputs=[X],
	updates=updates_stats
)

f_valid = theano.function(
	inputs=[X, targets],
	outputs=[loss_test, error_test],
)

best_valid_error = float('inf')
train_loss_hist, train_error_hist, valid_loss_hist, valid_error_hist, test_loss_hist, test_error_hist = [],[],[],[],[],[]
train_loss_100_hist = []

epoch = 0
patience = 5

while True:
	epoch += 1

	ei_train = train_stream.get_epoch_iterator()
	ei_valid = valid_stream.get_epoch_iterator()
	ei_test = test_stream.get_epoch_iterator()
	ei_train_100 = train_stream_100.get_epoch_iterator()

	train_loss, train_error, valid_loss, valid_error, test_loss, test_error, train_loss_100 = 0, 0, 0, 0, 0, 0, 0

	for batch_num in range(num_train_cifar100/batch_size):
		try:
			batch_100 = next(ei_train_100)
		except StopIteration:
			continue
		outputs = f_train_100(batch_100[1], batch_100[2])
		train_loss_100 += outputs[0]
			
	for batch_num in range(num_train_example/batch_size):
		try:
			batch = next(ei_train)
		except StopIteration:
			continue
		outputs = f_train(batch[0], batch[1])
		train_loss += outputs[0]
		train_error += outputs[1]

	# update statistics for batch norm
	ei_train = train_stream.get_epoch_iterator()
	for batch_num in range(num_train_example/batch_size):
		try:
			batch = next(ei_train)
		except StopIteration:
			continue

		f_stats(batch[0])

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
	train_loss_100 /= float(train_dataset_100.num_examples)/batch_size

	train_error_hist.append(train_error)
	train_loss_hist.append(train_loss)
	valid_error_hist.append(valid_error)
	valid_loss_hist.append(valid_loss)
	test_error_hist.append(test_error)
	test_loss_hist.append(test_loss)
	train_loss_100_hist.append(train_loss_100)

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


	# reinit batch accumulators:
	for val, acc in acc_parameters:
		acc.set_value(numpy.zeros(acc.get_value().shape))

# save train and valid hist
numpy.savez('plots_%s.npz' % (suffix,), [train_loss_hist, train_error_hist, valid_loss_hist, valid_error_hist, test_loss_hist, test_error_hist, train_loss_100_hist])
