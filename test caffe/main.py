import caffe
import cv2 
import numpy as np 
from PIL import Image
import cPickle as pickle 
from cifar_10_preprocess import  load_cifar10_data
from caffe.tools import lmdb_io
from caffe import layers as L
from caffe import params as P
import matplotlib.pyplot as plt

caffe.set_mode_gpu()
caffe.set_device(0)

train_net_path = 'model.prototxt'
train_lmdb = 'train_lmdb'
test_net_path = 'model_test.prototxt'
train_lmdb = 'test_lmdb'
solver_config_path = 'solver.prototxt'

def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    n.score = L.Softmax(n.ip2)
    return n.to_proto()

def train_accuracy():
    return sum(np.argmax(solver.net.blobs['score'].data,axis=1) == solver.net.blobs['label'].data)/100.0

def test_accuracy():
    return sum(np.argmax(solver.test_nets[0].blobs['score'].data,axis=1) == solver.test_nets[0].blobs['label'].data)/100.0

with open(train_net_path, 'w') as f:
    f.write(str(lenet(train_lmdb, 64)))
with open(test_net_path, 'w') as f:
    f.write(str(lenet(train_lmdb, 100)))


# train_data,train_label,test_data,test_label , label_names = load_cifar10_data('cifar-10-batches-py',channel_first=False , RGB_to_BGR=True)
# lmdb_io.LMDB('train_lmdb').write(train_data  , train_label)
# lmdb_io.LMDB('test_lmdb').write(test_data  , test_label)

solver = caffe.get_solver(solver_config_path)