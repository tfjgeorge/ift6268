from blocks_extras.extensions.plot import Plot

from blocks.algorithms import GradientDescent, Adam
from blocks.extensions import Printing, Timing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop

from fuel.datasets import CIFAR10
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from model import get_model
import numpy
import socket
import theano
from theano import tensor
from theano.tensor.nnet import categorical_crossentropy
from transformers import OneHotEncode, RandomHorizontalFlip


running_on_laptop = socket.gethostname() == 'yop'
if running_on_laptop:
	host_plot = 'http://localhost:5006'
	slice_train = slice(0, 32*10)
	slice_test = slice(32*10, 32*10+32*5)
else:
	host_plot = 'http://hades.calculquebec.ca:5042'
	slice_train = slice(0,45000-8)
	slice_test = slice(45000,50000-8)

batch_size = 32
num_train_example = slice_train.stop - slice_train.start
num_test_example = slice_test.stop - slice_test.start

train_dataset = CIFAR10(('train',), subset=slice_train)
train_stream = DataStream.default_stream(
    train_dataset,
    iteration_scheme=SequentialScheme(train_dataset.num_examples, batch_size)
)
train_stream = OneHotEncode(train_stream, which_sources=('targets',))
train_stream = RandomHorizontalFlip(train_stream, which_sources=('features',))

test_dataset = CIFAR10(('train',), subset=slice_test)
test_stream = DataStream.default_stream(
    test_dataset,
    iteration_scheme=SequentialScheme(test_dataset.num_examples, batch_size)
)
test_stream = OneHotEncode(test_stream, which_sources=('targets',))

X = tensor.ftensor4('features')
targets = tensor.fmatrix('targets')

output, output_test, all_parameters, acc_parameters = get_model(X, batch_size, (32, 32))

loss = categorical_crossentropy(output[:,:,0,0], targets).mean()
loss.name = 'loss'

loss_test = categorical_crossentropy(output_test[:,:,0,0], targets).mean()
loss.name = 'loss_test'

error = tensor.neq(tensor.argmax(output[:,:,0,0], axis=1), tensor.argmax(targets, axis=1)).mean()
error.name = 'error'

error_test = tensor.neq(tensor.argmax(output_test[:,:,0,0], axis=1), tensor.argmax(targets, axis=1)).mean()
error.name = 'error_test'

# construct update rule
learning_rate = 0.1
updates, updates_stats = [], []
for param in all_parameters:
	updates.append((param, param - learning_rate * tensor.grad(loss, param)))
for val, acc in acc_parameters:
	updates_stats.append((acc, acc + val/(num_train_example/batch_size)))

f_train = theano.function(
	inputs=[X, targets],
	outputs=[loss, error, acc_parameters[1][1].shape],
	updates=updates
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
train_loss_hist, train_error_hist, valid_loss_hist, valid_error_hist = [],[],[],[]

for epoch in range(20):
	ei_train = train_stream.get_epoch_iterator()
	ei_valid = test_stream.get_epoch_iterator()

	train_loss, train_error, valid_loss, valid_error = 0, 0, 0, 0


	for batch_num in range(num_train_example/batch_size):
		try:
			batch = next(ei_train)
		except StopIteration:
			continue
		outputs = f_train(batch[0], batch[1])
		train_loss += outputs[0]
		train_error += outputs[1]

		print outputs[2]

	# update statistics for batch norm
	ei_train = train_stream.get_epoch_iterator()
	for batch_num in range(num_train_example/batch_size):
		try:
			batch = next(ei_train)
		except StopIteration:
			continue

		f_stats(batch[0])

	for batch_num in range(num_test_example/batch_size):
		try:
			batch = next(ei_valid)
		except StopIteration:
			continue

		outputs = f_valid(batch[0], batch[1])
		valid_loss += outputs[0]
		valid_error += outputs[1]

	train_error /= num_train_example/batch_size
	train_loss /= num_train_example/batch_size
	valid_error /= num_test_example/batch_size
	valid_loss /= num_test_example/batch_size

	train_error_hist.append(train_error)
	train_loss_hist.append(train_loss)
	valid_error_hist.append(valid_error)
	valid_loss_hist.append(valid_loss)

	print 'epoch %i, train: %.2f, %.2f, valid: %.2f, %.2f' % (epoch+1, train_loss, train_error, valid_loss, valid_error)

	if valid_error < best_valid_error:
		best_valid_error = valid_error

		print 'improved valid error'
		# save weights
		numpy.savez('best_weights.npz', [param.get_value() for param in all_parameters])
		# save batch norm weights
		numpy.savez('best_weights_bn.npz', [acc.get_value() for val, acc in acc_parameters])


	# reinit batch accumulators:
	for val, acc in acc_parameters:
		acc.set_value(numpy.zeros(acc.get_value().shape))

# save train and valid hist
numpy.savez('plots.npz', [train_loss_hist, train_error_hist, valid_loss_hist, valid_error_hist])
