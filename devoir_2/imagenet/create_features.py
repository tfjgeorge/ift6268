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
train_stream = ScaleAndShift(train_stream, 1., 0, which_sources=('features',))
train_stream = Cast(train_stream, 'floatX', which_sources=('features',))

valid_dataset = CIFAR10(('train',), subset=slice_valid)
valid_stream = DataStream.default_stream(
    valid_dataset,
    iteration_scheme=SequentialScheme(valid_dataset.num_examples, batch_size)
)
valid_stream = OneHotEncode10(valid_stream, which_sources=('targets',))
valid_stream = MinimumImageDimensions(valid_stream, (224, 224), which_sources=('features',))
valid_stream = ScaleAndShift(valid_stream, 1., 0, which_sources=('features',))
valid_stream = Cast(valid_stream, 'floatX', which_sources=('features',))


test_dataset = CIFAR10(('train',), subset=slice_test)
test_stream = DataStream.default_stream(
    test_dataset,
    iteration_scheme=SequentialScheme(test_dataset.num_examples, batch_size)
)
test_stream = OneHotEncode10(test_stream, which_sources=('targets',))
test_stream = MinimumImageDimensions(test_stream, (224, 224), which_sources=('features',))
test_stream = ScaleAndShift(test_stream, 1., 0, which_sources=('features',))
test_stream = Cast(test_stream, 'floatX', which_sources=('features',))

## build computational graph
X = tensor.ftensor4('features')
targets = tensor.fmatrix('targets')

net = build_model()
# loading pretrained weights
model = pickle.load(open('./blvc_googlenet.pkl'))
lasagne.layers.set_all_param_values(net['prob'], model['param values'])

googlenet_features = lasagne.layers.get_output(net['pool5/7x7_s1'], X)

f_features = theano.function(
	inputs=[X],
	outputs=[googlenet_features]
)

ei_train = train_stream.get_epoch_iterator()
ei_valid = valid_stream.get_epoch_iterator()
ei_test = test_stream.get_epoch_iterator()

features_train, features_valid, features_test = [], [], []
targets_train, targets_valid, targets_test = [], [], []
		
for batch_num in range(num_train_example/batch_size):
	try:
		batch = next(ei_train)
	except StopIteration:
		continue

	outputs = f_features(batch[0])
	features_train.append(outputs[0])
	targets_train.append(batch[1])

for batch_num in range(num_valid_example/batch_size):
	try:
		batch = next(ei_valid)
	except StopIteration:
		continue

	outputs = f_features(batch[0])
	features_valid.append(outputs[0])
	targets_valid.append(batch[1])

for batch_num in range(num_test_example/batch_size):
	try:
		batch = next(ei_test)
	except StopIteration:
		continue

	outputs = f_features(batch[0])
	features_test.append(outputs[0])
	targets_test.append(batch[1])

# save train and valid hist
numpy.savez_compressed('features_compressed.npz',
	features_train=features_train,
	targets_train=targets_train,
	features_valid=features_valid,
	targets_valid=targets_valid,
	features_test=features_test,
	targets_test=targets_test
)
