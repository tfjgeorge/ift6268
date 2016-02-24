from blocks.extensions import Printing, Timing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.algorithms import GradientDescent, Adam, Scale
from blocks.main_loop import MainLoop
from blocks_extras.extensions.plot import Plot

from fuel.datasets import CIFAR10
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from model import get_model
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

output, output_test, all_parameters = get_model(X, batch_size, 32)

loss = categorical_crossentropy(output[:,:,0,0], targets).mean()
loss.name = 'loss'

error = tensor.neq(tensor.argmax(output[:,:,0,0], axis=1), tensor.argmax(targets, axis=1)).mean()
error.name = 'error'

algorithm = GradientDescent(
	cost=loss,
	parameters=all_parameters,
#	step_rule=Adam(),
	step_rule=Scale(learning_rate=0.05),
	on_unused_sources='ignore'
)

extensions = [
	Timing(),
	TrainingDataMonitoring([loss, error], after_epoch=True),
	DataStreamMonitoring(variables=[loss, error], data_stream=test_stream, prefix="test"),
	# Plot('CIFAR10', channels=[['loss','test_loss']], after_epoch=True, server_url=host_plot),
	Plot('CIFAR10', channels=[['loss','test_loss'],['error','test_error']], after_epoch=True, server_url=host_plot),
	Printing(),
	FinishAfter(after_n_epochs=20),
	Checkpoint('cifar_train')
]

main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                     extensions=extensions)
main_loop.run()
