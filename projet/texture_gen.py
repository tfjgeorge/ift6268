import theano
import numpy
from theano import tensor
theano.config.floatX = 'float32'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASEPATH = '/tmp/textureexps/pepper1/'
fig_n = 0

import lasagne
import pickle

from vggnet import build_model
vggnet_model = build_model()

params_val = pickle.load(open('./vgg19.pkl'))
lasagne.layers.set_all_param_values(vggnet_model['pool5'], params_val['param values'][:32])



X = tensor.ftensor4('image')

Z_names = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
Zs = dict()
for i in range(5):
    Zs['Z%d' % (i+1,)] = tensor.ftensor4('Z%d' % (i+1,))



from tg_model_bricks import build_model

generated_image_graph = build_model(Zs)



f_texture_generation = theano.function([Zs[n] for n in Z_names], [generated_image_graph])



# test texture generation net
theano.config.exception_verbosity = 'low'

def generate_noisy_images(n=1):
    Z1 = numpy.random.rand(n, 3, 256, 256).astype('float32')*2-1
    Z2 = numpy.random.rand(n, 3, 128, 128).astype('float32')*2-1
    Z3 = numpy.random.rand(n, 3, 64, 64).astype('float32')*2-1
    Z4 = numpy.random.rand(n, 3, 32, 32).astype('float32')*2-1
    Z5 = numpy.random.rand(n, 3, 16, 16).astype('float32')*2-1
    
    return Z1, Z2, Z3, Z4, Z5

Z1, Z2, Z3, Z4, Z5 = generate_noisy_images()

o = f_texture_generation(Z1, Z2, Z3, Z4, Z5)
plt.imshow(numpy.rollaxis(o[0][0], 0, 3)/numpy.max(o[0][0]))



texture_image = mpimg.imread('./red-peppers256.jpg')
plt.imshow(texture_image)

print texture_image.min(), texture_image.max()



def gram_matrix(X):
    X = X.flatten(ndim=3)
    return tensor.batched_tensordot(X, X, axes=[[2],[2]])



texture_loss_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

def texture_features(X):
    layers = [vggnet_model[k] for k in texture_loss_layers]
    outputs = lasagne.layers.get_output(layers, X)
    
    return outputs



MEAN_VALUES = numpy.array([104, 117, 123]).reshape((3,1,1))


texture_image_nn_input = numpy.rollaxis(texture_image, 2)[None]
# Convert RGB to BGR
texture_image_nn_input = texture_image_nn_input[::-1, :, :]-MEAN_VALUES
texture_image_nn_input = texture_image_nn_input.astype('float32')

# print texture_image_nn_input
print texture_image_nn_input.shape

f_features_gram = theano.function(
        inputs=[X],
        outputs=[gram_matrix(f) for f in texture_features(X)]
)
target_image_features = f_features_gram(texture_image_nn_input)
# print target_image_features
print [t.shape for t in target_image_features]

from blocks.graph import ComputationGraph, apply_batch_normalization, get_batch_normalization_updates

cg = ComputationGraph(generated_image_graph)
cg_bn = apply_batch_normalization(cg)
pop_updates = get_batch_normalization_updates(cg_bn)



text_generated = texture_features(cg.outputs[0])
gram_generated = [gram_matrix(f) for f in text_generated]

loss = 0
for i in range(len(target_image_features)):
    N = text_generated[i].shape[1]
    M = text_generated[i].shape[2]*text_generated[i].shape[3]
    loss += 1./ (4 * 16 * N ** 2 * M ** 2) * ((gram_generated[i]
        - tensor.addbroadcast(theano.shared(target_image_features[i]), 0)) ** 2).sum()


alpha = 0.1
extra_updates = [(p, m * alpha + p * (1 - alpha))
    for p, m in pop_updates]



from fuel.datasets import Dataset

class RandomImagesDataset(Dataset):
    
    def __init__(self, sources=None):
        self.provides_sources = ('Z1', 'Z2', 'Z3', 'Z4', 'Z5')
        if sources == None:
            sources = self.provides_sources
        super(RandomImagesDataset, self).__init__(sources=sources)
        
    def get_data(self, state, request):
        n = len(request)
        Z1 = numpy.random.rand(n, 3, 256, 256).astype('float32')*2-1
        Z2 = numpy.random.rand(n, 3, 128, 128).astype('float32')*2-1
        Z3 = numpy.random.rand(n, 3, 64, 64).astype('float32')*2-1
        Z4 = numpy.random.rand(n, 3, 32, 32).astype('float32')*2-1
        Z5 = numpy.random.rand(n, 3, 16, 16).astype('float32')*2-1
        
        return [Z1, Z2, Z3, Z4, Z5]



from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

ds = RandomImagesDataset()
train_stream = DataStream.default_stream(
    ds,
    iteration_scheme=SequentialScheme(140, 14)
)



from blocks.algorithms import GradientDescent, Adam, Scale, Momentum
adaminitlr = 1e-3 # last run 3 (before 5)

algorithm = GradientDescent(cost=loss, parameters=cg.parameters,
    step_rule=Adam(adaminitlr)) # 1 6 11 4
#    step_rule=Adam(1e-20))
#    step_rule=Scale(1e-5))
# 1 11 5                           
#    step_rule=Momentum(1e-5))
       
algorithm.add_updates(extra_updates)




# algorithm.step_rule.learning_rate.set_value(1e-1)

# main_loop.algorithm.step_rule = Adam(1e-3)

def savefig():
    global fig_n
    plt.savefig(BASEPATH+str(fig_n)+'.png')
    print 'saved fig %d.png' % (fig_n, )
    fig_n += 1

def print_img():
    Z1, Z2, Z3, Z4, Z5 = generate_noisy_images(16)

    rgb = -1

    o = f_texture_generation(Z1, Z2, Z3, Z4, Z5)
    img = numpy.rollaxis(o[0][0]+MEAN_VALUES, 0, 3)[::rgb,:,:]
    img = numpy.clip(img, 0, 255).astype('uint8')
    plt.imshow(img)
    savefig()
    img = numpy.rollaxis(o[0][1]+MEAN_VALUES, 0, 3)[::rgb,:,:]
    img = numpy.clip(img, 0, 255).astype('uint8')
    plt.imshow(img)
    savefig()


from blocks.extensions import SimpleExtension

class PrintImageExtension(SimpleExtension):
    
    def do(self, which_callback, *args):
        print_img()



from blocks.extensions import Printing, Timing
from blocks.extensions.training import TrackTheBest
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks_extras.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint

import datetime

loss.name = 'loss'

extensions = [
    Timing(),
    TrainingDataMonitoring([loss], after_epoch=True),
    Plot('FF text gen %s' % (datetime.datetime.now(), ),
         channels=[['loss']], after_batch=True),
    TrackTheBest('loss'),
    FinishIfNoImprovementAfter('loss_best_so_far', epochs=5),
    Printing(),
    PrintImageExtension(every_n_epochs=5),
    Checkpoint(BASEPATH+'model.pkl')
]



from blocks.model import Model

model = Model(generated_image_graph)



from blocks.main_loop import MainLoop


main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                     extensions=extensions, model=model)


main_loop.run()



# reinit adam
main_loop.algorithm.step_rule = Adam(adaminitlr)
extensions[4] = FinishIfNoImprovementAfter('loss_best_so_far', epochs=20)
extensions[4].main_loop = main_loop

for i in range(5):
    
    # print_img()
    new_lr = 0.2*algorithm.step_rule.learning_rate.get_value()
    print '===\n(%d) Learning rate set to %e\n===' % (i, new_lr)
    
    # 
    algorithm.step_rule.learning_rate.set_value(
        numpy.float32(new_lr))
    #main_loop.algorithm.step_rule = Adam(new_lr)
    
    # reinit early stopping
    extensions[4].last_best_iter = main_loop.log.status['iterations_done']
    extensions[4].last_best_epoch = main_loop.log.status['epochs_done']

    main_loop.run()



print_img()



Z1, Z2, Z3, Z4, Z5 = generate_noisy_images(16)

rgb = -1

o = f_texture_generation(Z1, Z2, Z3, Z4, Z5)

plt.imshow((numpy.rollaxis(o[0][0], 0, 3)[::rgb,:,:]
            -numpy.min(o[0][0]))/(numpy.max(o[0][0])-numpy.min(o[0][0])))
savefig()
plt.imshow((numpy.rollaxis(o[0][1], 0, 3)[::rgb,:,:]
            -numpy.min(o[0][1]))/(numpy.max(o[0][1])-numpy.min(o[0][1])))

print numpy.min(o[0][0])
print numpy.max(o[0][0])

print numpy.max(o[0][0]+MEAN_VALUES)
print numpy.min(o[0][0]+MEAN_VALUES)




plt.imshow(numpy.rollaxis(o[0][0], 0, 3)+texture_image.mean())




import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



Z1, Z2, Z3, Z4, Z5 = generate_noisy_images(2)



f = plt.figure(frameon=False, figsize=(10, 10), dpi=100)
canvas_width, canvas_height = f.canvas.get_width_height()
ax = f.add_axes([0, 0, 1, 1])
ax.axis('off')

outf = '/tmp/test.avi'
rate = 24

cmdstring = ('/usr/bin/ffmpeg', 
    '-y', '-r', '30', # overwrite, 30fps
    '-s', '%dx%d' % (canvas_width, canvas_height), # size of image string
    '-pix_fmt', 'argb', # format
    '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
    '-vcodec', 'mpeg4', outf)
p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

for i in numpy.arange(0, 1, 0.01):
    z1 = Z1[0:1]*i + (1-i)*Z1[1:2]
    z2 = Z2[0:1]*i + (1-i)*Z2[1:2]
    z3 = Z3[0:1]*i + (1-i)*Z3[1:2]
    z4 = Z4[0:1]*i + (1-i)*Z4[1:2]
    z5 = Z5[0:1]*i + (1-i)*Z5[1:2]

    rgb = -1

    o = f_texture_generation(z1, z2, z3, z4, z5)
    img = numpy.rollaxis(o[0][0]+MEAN_VALUES, 0, 3)[::rgb,:,:]
    img = numpy.clip(img, 0, 255).astype('uint8')
            
    plt.imshow(img)
    plt.draw()
    # extract the image as an ARGB string
    string = f.canvas.tostring_argb()

    # write to pipe
    p.stdin.write(string)

p.stdin.close()
