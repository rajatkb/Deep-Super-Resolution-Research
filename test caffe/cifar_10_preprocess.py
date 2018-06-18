import numpy as np 
from PIL import Image
import cPickle as pickle 

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def load_cifar10_data(data_dir , channel_first=True , RGB_to_BGR=False):
    train_data = None
    train_labels = []
    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data']
            train_labels = data_dic['labels']
        else:
            train_data = np.vstack((train_data, data_dic['data']))
            train_labels += data_dic['labels']

    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic['data']
    test_labels = test_data_dic['labels']

    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    if not channel_first:
        train_data = np.rollaxis(train_data, 1, 4)
    train_labels = np.array(train_labels)

    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    if not channel_first:
        test_data = np.rollaxis(test_data, 1, 4)
    test_labels = np.array(test_labels)

    data_dic=unpickle(data_dir+"/batches.meta")

    if RGB_to_BGR:
        train_data = train_data[:,...,::-1]
        test_data = test_data[:,...,::-1]
        
    return train_data, train_labels, test_data, test_labels , data_dic['label_names']