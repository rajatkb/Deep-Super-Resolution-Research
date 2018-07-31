import cv2
import os

import argparse
parser = argparse.ArgumentParser(description='control SRCNN')
parser.add_argument('--to', action="store",dest="tryout", default=20000)
parser.add_argument('--ep', action="store",dest="epochs", default=50)
parser.add_argument('--bs', action="store",dest="batch_size", default=128)
parser.add_argument('--lr', action="store",dest="learning_rate", default=0.001)
parser.add_argument('--gpu', action="store",dest="gpu", default=1)
parser.add_argument('--test_sample', action="store",dest="test_sample",default=190)
parser.add_argument('--scale', action='store' , dest = 'downscale' , default = 2)

values = parser.parse_args()
learning_rate = float(values.learning_rate)
batch_size = int(values.batch_size)
epochs = int(values.epochs)
tryout = int(values.tryout)
gpu=values.gpu
test_sample = int(values.test_sample)
downscale = int(values.downscale)
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

import sys
import numpy as np
import matplotlib.pyplot as plt
from SIBLING_DATA import PARSE_DATA
from keras.models import Model
from keras.layers import Input, Convolution2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras import backend as k
from keras.models import load_model
import tensorflow as tf

folder = '../sibling-face-data/DBs'
HEIGHT = 512
WIDTH = 512
CHANNEL = 3
# DATA = PARSE_DATA(folder = "../sibling-face-data/DBs", WIDTH=512 , HEIGHT=512)

patch_size = 64 # DATA.patch_hr_size


def PSNRLoss(y_true, y_pred):
    return 10* k.log(255**2 /k.mean(k.square(y_pred - y_true)))

def PSNRLossnp(y_true,y_pred):
    return 10* np.log(255*2 / np.mean(np.square(y_pred - y_true)))

def SSIM(y_true,y_pred):
    u_true = k.mean(y_true)
    u_pred = k.mean(y_pred)
    var_true = k.var(y_true)
    var_pred = k.var(y_pred)
    std_true = k.sqrt(var_true)
    std_pred = k.sqrt(var_pred)
    c1 = k.square(0.01*7)
    c2 = k.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom



def patchify( img  , scale = 1):
            p = int(64/scale)
            r = int(img.shape[0] / p)
            c = int(img.shape[1] / p)
            patch_list = []
            for R in range(0,r):
                for C in range(0,c):
                    patch_list.append(img[R*p:(R+1)*p,C*p:(C+1)*p])
            return patch_list,r,c

def reconstruct( li ,r , c , scale=1):
            image = np.zeros((int(r*(64/scale)),int(c*(64/scale)) , 3))
            print(image.shape)
            i = 0
            p = int(64 / scale)
            for R in range(0,r):
                for C in range(0,c):
                    image[R*p:(R+1)*p,C*p:(C+1)*p] = li[i]
                    i = i+1
            return np.array(image, np.uint8)



class SRCNN:
    def __init__(self,width , height , channel , lr , load_from_disk=True):
        self.height = height
        self.width = width
        if load_from_disk == True:
            self.model = load_model('SRCNN_sibling_x'+str(downscale)+'.h5', custom_objects={'SSIM':SSIM , 'PSNRLoss':PSNRLoss})
        else:    
            inputs = Input(shape=(height, width , channel))
            x = Convolution2D(64, (9, 9), activation='relu', init='he_normal' , border_mode='same')(inputs)
            x = Convolution2D(32, (1, 1), activation='relu', init='he_normal' , border_mode='same')(x)
            x = Convolution2D(3, (5, 5), init='he_normal' , border_mode='same')(x)
            self.model = Model(input = inputs, output = x)
        adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
        self.model.compile(loss='mse', optimizer=adam , metrics=[PSNRLoss , SSIM])
        self.model.summary()
        input()

    def fit(self , X , Y ,batch_size=64 , epoch = 100 , gpu=0 ):
        # with tf.device('/gpu:'+str(gpu)):    
        hist = self.model.fit(X, Y , batch_size = batch_size , verbose =1 , nb_epoch=epoch)
        return hist.history
        
    def get_model(self):
        return self.model

    def generate(self , X ):
        return self.model.predict(X)
            
from PIL import Image

# X,Xb,Y = DATA.get_train_patchwise_data(downscale = downscale , noisy=True)
# X_test,Xb_test,Y_test = DATA.get_train_patchwise_data(downscale = downscale , noisy=True)
## No noisy image was put in the original training set
## we put noisy image in order to test it against noisy enivornments



        
        
        

model = SRCNN(patch_size , patch_size , CHANNEL , learning_rate)

img= plt.imread('Set5/baby.png')
im = cv2.resize(img , (int(img.shape[1]/2) , int(img.shape[0]/2)) , interpolation=cv2.INTER_CUBIC)
im = cv2.resize(im , (int(im.shape[1]*2) , int(im.shape[0]*2)) , interpolation=cv2.INTER_CUBIC)
print(im.shape , img.shape)

im = patchify(im)[0]
img = patchify(img)[0]

model.fit( np.array(im), np.array(img) , epoch = epochs , batch_size=batch_size)

result = model.generate(np.array(im))
result = reconstruct( result , 8 , 8 , scale=1)

Image.fromarray(result).save('baby_2x.png')

# print(PSNRLossnp( result , im))

# original_img_patch = Y_test[test_sample*DATA.patch_count:(test_sample+1)*DATA.patch_count]
# lr_bicubic_patch= Xb_test[test_sample*DATA.patch_count:(test_sample+1)*DATA.patch_count]
# lr_img_patch = X_test[test_sample*DATA.patch_count:(test_sample+1)*DATA.patch_count]

# original_img = DATA.reconstruct(original_img_patch,downscale = downscale, var='hr')
# lr_bicubic = DATA.reconstruct(lr_bicubic_patch,downscale = downscale , var='hr')
# lr_img = DATA.reconstruct(lr_img_patch , downscale = downscale , var = 'lr')

# Image.fromarray(original_img).save("test_image.jpeg")
# Image.fromarray(lr_bicubic).save("test_image_2x.jpeg")
# Image.fromarray(lr_img).save("test_image_lr.jpeg")


# import time
# file = open("metrics.txt",'w') 
# start = time.time()
# for i in range (tryout):
#     print("tryout no: ",i)
#     hist = model.fit( Xb, Y , epoch = epochs , batch_size=batch_size)
#     elapse = start - time.time()
    
#     im = model.generate(Xb_test[test_sample*DATA.patch_count:(test_sample+1)*DATA.patch_count])
#     im = DATA.reconstruct(im,downscale = downscale, var='hr')
#     im[im > 255] = 255
#     im[im < 0] = 0
    
#     psnr = PSNRLossnp(original_img , im)
#     print("PSNR for reconstrcted image is : ",psnr)
    
    
#     file.write("Super Epoch (tryout):"+str(i)+" loss:"+str(hist['loss'][0])+" PSNRLoss:"+str(hist['PSNRLoss'][0])+" SSIM loss: "+str(hist['SSIM'][0]) +" PSNR for reconstruction: "+str(psnr)+" \n")
    
#     Image.fromarray(im.astype('uint8')).save("gen_image_2x_"+str(i)+"_.png")
#     model.get_model().save('SRCNN_sibling_x'+str(downscale)+'.h5')
    
# close(file)        
## already trained 850 passes 