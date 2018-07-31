import argparse
parser = argparse.ArgumentParser(description='control RDNSR')
parser.add_argument('--gpu', action="store",dest="gpu", default=-1)
parser.add_argument('--chk',action="store",dest="chk",default=-1)
# parser.add_argument('--test_sample', action="store",dest="test_sample",default=190)
parser.add_argument('--scale', action='store' , dest = 'scale' , default = 2)
parser.add_argument('--test_image', action = 'store' , dest = 'test_image' , default = 'test.png')
#parser.add_argument('--progressive_zoom',action='store',dest = 'pzoom' , default = False)


values = parser.parse_args()
gpu=int(values.gpu)
# test_sample = int(values.test_sample)
scale = int(values.scale)
chk = int(values.chk)
test_image = values.test_image

if gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

import sys
import numpy as np
import matplotlib.pyplot as plt
from SRIP_DATA_BUILDER import DATA
from keras.models import Model
from keras.layers import Input,MaxPool2D,Deconvolution2D ,Convolution2D , Add, Dense , AveragePooling2D , UpSampling2D , Reshape , Flatten , Subtract , Concatenate
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras import backend as k
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.utils import plot_model
from subpixel import Subpixel
from PIL import Image
import cv2


class SRResnet:
    def L1_loss(self , y_true , y_pred):
        return k.mean(k.abs(y_true - y_pred))
    
    #def L1_plus_PSNR_loss(self,y_true, y_pred):
        #return self.L1_loss(y_true , y_pred) - 0.0001*PSNRLoss(y_true , y_pred)
    
    def RDBlocks(self,x,name , count = 6 , g=32):
        ## 6 layers of RDB block
        ## this thing need to be in a damn loop for more customisability
            li = [x]
            pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu' , name = name+'_conv1')(x)
            
            for i in range(2 , count+1):
                li.append(pas)
                out =  Concatenate(axis = self.channel_axis)(li) # conctenated out put
                pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu', name = name+'_conv'+str(i))(out)
            
            # feature extractor from the dense net
            li.append(pas)
            out = Concatenate(axis = self.channel_axis)(li)
            feat = Convolution2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='same',activation='relu' , name = name+'_Local_Conv')(out)
            
            feat = Add()([feat , x])
            return feat
        
    def visualize(self):
            plot_model(self.model, to_file='model.png' , show_shapes = True)
    
    def get_model(self):
        return self.model
    
    def __init__(self , channel = 3 , lr=0.0001 , patch_size=32 , RDB_count=20 ,chk = -1 , scale = 2):
            self.channel_axis = 3
            inp = Input(shape = (patch_size , patch_size , channel))

            pass1 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(inp)

            pass2 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(pass1)

            
            RDB = self.RDBlocks(pass2 , 'RDB1')
            RDBlocks_list = [RDB,]
            for i in range(2,RDB_count+1):
                RDB = self.RDBlocks(RDB ,'RDB'+str(i))
                RDBlocks_list.append(RDB)
            out = Concatenate(axis = self.channel_axis)(RDBlocks_list)
            out = Convolution2D(filters=64 , kernel_size=(1,1) , strides=(1,1) , padding='same')(out)
            out = Convolution2D(filters=64 , kernel_size=(3,3) , strides=(1,1) , padding='same')(out)

            output = Add()([out , pass1])
            
            if scale >= 2:
                output = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(output)
            if scale >= 4:
                output = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(output)
            if scale >= 8:
                output = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(output)
            
            output = Convolution2D(filters =3 , kernel_size=(3,3) , strides=(1 , 1) , padding='same')(output)

            model = Model(inputs=inp , outputs = output)
            adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=lr/2, amsgrad=False)
            
            ## multi gpu setting
            
            #if gpu < 0:
               #model = multi_gpu_model(model, gpus=2)
            ## Modification of adding PSNR as a loss factor
            model.compile(loss=self.L1_loss, optimizer=adam , metrics=[PSNRLoss,SSIM])
            
            if chk > 0 :
        	print("Give a defined checkpoint to load from!!!")
		exit()

	    print("loading existing weights !!!")        	
	    model.load_weights('model_'+str(scale)+'x_iter'+str(chk)+'.h5')
            self.model = model
            
    def fit(self , X , Y ,batch_size=16 , epoch = 1000 ):
            # with tf.device('/gpu:'+str(gpu)):    
            hist = self.model.fit(X, Y , batch_size = batch_size , verbose =1 , nb_epoch=epoch)
            return hist.history



