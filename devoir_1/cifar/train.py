from fuel.datasets import CIFAR10
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from model import get_model
import theano
from theano import tensor
from transformers import OneHotEncode, RandomHorizontalFlip
from theano.tensor.nnet import categorical_crossentropy

from blocks.extensions import Printing, Timing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.algorithms import GradientDescent, Adam
from blocks.main_loop import MainLoop
from blocks_extras.extensions.plot import Plot


batch_size = 32

train_dataset = CIFAR10(('train',), subset=slice(0,45000))
train_stream = DataStream.default_stream(
    train_dataset,
    iteration_scheme=SequentialScheme(train_dataset.num_examples, batch_size)
)
train_stream = OneHotEncode(train_stream, which_sources=('targets',))
train_stream = RandomHorizontalFlip(train_stream, which_sources=('features',))

test_dataset = CIFAR10(('train',), subset=slice(45000,50000))
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

algorithm = GradientDescent(
	cost=loss,
	parameters=all_parameters,
	step_rule=Adam(),
	on_unused_sources='ignore'
)

host_plot = 'http://localhost:5006'

extensions = [
	Timing(),
	TrainingDataMonitoring([loss], after_epoch=True),
	DataStreamMonitoring(variables=[loss], data_stream=test_stream, prefix="test"),
	Plot('CIFAR10', channels=[['loss','test_loss']], after_epoch=True, server_url=host_plot),
	Printing(),
	FinishAfter(after_n_epochs=20)
]

main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                     extensions=extensions)
main_loop.run()
