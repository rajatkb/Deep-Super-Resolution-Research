import cv2
import os

'''
train where the train data is build somewhere and you are pointing to it
python main.py --ep 1 --to 200 --bs 16 --lr 0.0001 --gpu 1 --sample 16384 --chk 159 --scale 2 --data ../Data --test_image div2k_test.png

testing on the data

python main.py --test_only True --gpu 1 --chk 159 --scale 2 --test_image some.png

use the above to run the file. this is the orinal configuration as per the paper

'''

import argparse
parser = argparse.ArgumentParser(description='control RDNSR')
parser.add_argument('--to', action="store",dest="tryout", default=200)
parser.add_argument('--ep', action="store",dest="epochs", default=1000)
parser.add_argument('--bs', action="store",dest="batch_size", default=16)
parser.add_argument('--lr', action="store",dest="learning_rate", default=0.0001)
parser.add_argument('--gpu', action="store",dest="gpu", default=-1)
parser.add_argument('--chk',action="store",dest="chk",default=-1)
parser.add_argument('--sample',action='store',dest="sample",default=512)
# parser.add_argument('--test_sample', action="store",dest="test_sample",default=190)
# parser.add_argument('--scale', action='store' , dest = 'scale' , default = 2)
parser.add_argument('--data', action='store' , dest = 'folder' , default = '../Data')
parser.add_argument('--test_image', action = 'store' , dest = 'test_image' , default = 'test.png')
parser.add_argument('--test_only' , action = 'store', dest = 'test_only' , default = False)
parser.add_argument('--zoom',action='store',dest = 'zoom' , default = False)

parser.add_argument('--content',action = 'store',dest = 'content_lambda' , default = 1)
parser.add_argument('--pix' , action = 'store' , dest = 'mse_lambda' , default = 1)
parser.add_argument('--adv' , action = 'store' , dest='adv_lambda' , default = 1)

values = parser.parse_args()
learning_rate = float(values.learning_rate)
batch_size = int(values.batch_size)
epochs = int(values.epochs)
tryout = int(values.tryout)
gpu=int(values.gpu)
sample = int(values.sample)
# test_sample = int(values.test_sample)
# scale = int(values.scale)
folder = values.folder
test_only = values.test_only
chk = int(values.chk)
zoom = values.zoom
content_lambda = float(values.content_lambda)
mse_lambda = float(values.mse_lambda)
adv_lambda = float(values.adv_lambda)

if gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)



import sys
import numpy as np
import matplotlib.pyplot as plt
from SRIP_DATA_BUILDER import DATA
from keras.models import Model
from keras.layers import *
from keras.callbacks import LearningRateScheduler
from keras import backend as k
from keras.applications import vgg16
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
import tensorflow as tf
from keras.utils import plot_model
from subpixel import Subpixel
from PIL import Image
import cv2




def PSNRLossnp(y_true,y_pred):
        return 10* np.log(255*2 / (np.mean(np.square(y_pred - y_true))))

def SSIM( y_true,y_pred):
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

def PSNRLoss(y_true, y_pred):
        return 10* k.log(255**2 /(k.mean(k.square(y_pred - y_true))))


class SRResnet:

    def MSE(self , y_true , y_pred):
        return k.mean(k.square(y_true - y_pred))

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
            pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu', name = name+'_sep_conv'+str(i))(out)
            
        # feature extractor from the dense net
        li.append(pas)
        out = Concatenate(axis = self.channel_axis)(li)
        feat = Convolution2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='same',activation='relu' , name = name+'_Local_Conv')(out)
        
        feat = Add()([feat , x])
        return feat
        
    def visualize(self):
        plot_model(self.modelx_2x, to_file='model1.png' , show_shapes = True)
        plot_model(self.model2x_4x, to_file='model2.png' , show_shapes = True)
        plot_model(self.model4x_8x, to_file='model3.png' , show_shapes = True)
        plot_model(self.inference_model, to_file='model.png' , show_shapes = True)
    
    def get_model(self):
        return self.inference_model

    def Adversarial_Loss(self , scale):
        inp = Input(shape = (self.patch_size*scale , self.patch_size*scale , self.channel_axis))
        x = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(inp)
        x = self.RDBlocks(x , 'layer1' , count = 6 , g = 32)
        x = self.RDBlocks(x , 'layer2' , count = 6 , g = 32)
        x = Convolution2D(filters=64, kernel_size=(3,3), strides=(2, 2), padding='valid' , activation='relu')(x)
        x = Convolution2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(x)
        x = Convolution2D(filters=128, kernel_size=(3,3), strides=(2, 2), padding='valid' , activation='relu')(x)
        x = Dropout(0.50)(x)
        x = Convolution2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(x)
        x = Convolution2D(filters=256, kernel_size=(3,3), strides=(2, 2), padding='valid' , activation='relu')(x)
        x = Dropout(0.20)(x)
        x = Convolution2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(x)
        x = Convolution2D(filters=512, kernel_size=(3,3), strides=(2, 2), padding='valid' , activation='relu')(x)
        x = Dropout(0.10)(x)
        x = Flatten()(x)
        x = Dense(32 , activation='relu')(x)
        x = Dense(16  , activation='relu')(x)
        x = Dense(1 , activation='sigmoid' , name='discriminator_'+str(scale)+'x')(x)
        discriminator = Model(inputs = inp , outputs = x)
        return discriminator



    def VGG_content_loss(self , x , inp , scale ):


        vgg_inp = Input(shape = (self.patch_size*scale, self.patch_size*scale, self.channel_axis))
        vgg = vgg16.VGG16(input_tensor = vgg_inp , weights='imagenet' , include_top=False)
        for l in vgg.layers: l.trainable=False

        self.content_loss_layers = ['block3_conv3','block4_conv3']
        content_loss_output = [vgg.get_layer(l).output for l in self.content_loss_layers]
        vgg_content_model = Model(inputs=vgg_inp , outputs = content_loss_output)

        hr_vgg = vgg_content_model(inp) ## list of output
        pred_vgg = vgg_content_model(x) ## list of output

        content_losses = []
        for i in range(len(self.content_loss_layers)):
            content_losses.append(Lambda(self.get_content_loss , output_shape=(1,))([pred_vgg[i],hr_vgg[i]]))
        combined_content_loss = Add()(content_losses)
        # k.constant(10**(scale - 8))*
        combined_content_loss = Lambda(lambda x:x , name='content_loss_'+str(scale)+'x')(combined_content_loss) 

        return combined_content_loss

    def get_content_loss(self,args):
        new_activation, content_activation = args[0], args[1]
        sq = k.square(new_activation - content_activation)
        out = k.mean(sq , axis=[1,2,3])
        return  out

    def get_RDN_pass(self, scale , depth=3 ):
        inp = Input(shape = ((scale // 2) * self.patch_size , (scale // 2) * self.patch_size , self.channel_axis))
        pass1 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(inp)
        pass2 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(pass1)
        
        RDB = self.RDBlocks(pass2 , 'RDB1_'+str(scale))
        RDBlocks_list = [RDB,]
        for i in range(2,depth+1):
            RDB = self.RDBlocks(RDB ,'RDB'+str(i)+'_'+str(scale))
            RDBlocks_list.append(RDB)
        out = Concatenate(axis = self.channel_axis)(RDBlocks_list)
        out = Convolution2D(filters=64 , kernel_size=(1,1) , strides=(1,1) , padding='same')(out)
        out = Convolution2D(filters=64 , kernel_size=(3,3) , strides=(1,1) , padding='same')(out)
        output = Add()([out , pass1])
        output = Conv2DTranspose(filters=64 , kernel_size=3  , strides=(2,2), padding='same')(output)
        output = Reshape((self.patch_size*scale , self.patch_size*scale , 64))(output)
        output = Convolution2D(filters =32 , kernel_size=(1,1) , strides=(1 , 1) , activation='relu' , padding='same')(output)
        output = Convolution2D(filters =32 , kernel_size=(3,3) , strides=(1 , 1) , activation='relu' , padding='same')(output)
        output = Convolution2D(filters =3 , kernel_size=(3,3) , strides=(1 , 1) , padding='same' , activation='sigmoid', name='output'+str(scale)+'x')(output)

        return Model(inputs = inp , outputs = output )
    
    def __init__(self , content_lambda , mse_lambda , adv_lambda , channel = 3 , lr=0.0001 , patch_size=32 ,chk = -1 , test = False ,visualize=True ):
        self.channel_axis = 3
        self.patch_size = patch_size
        self.content_lambda = content_lambda
        self.mse_lambda = mse_lambda
        self.call_count = 1
        self.adv_lambda = adv_lambda

        self.adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=lr/10, amsgrad=False)

        inp = Input(shape = (patch_size , patch_size , channel) , name='inputx')

        inp2x = Input(shape = (patch_size*2 , patch_size*2 , channel) , name='input2x')
        inp4x = Input(shape = (patch_size*4 , patch_size*4 , channel) , name='input4x')
        inp8x = Input(shape = (patch_size*8 , patch_size*8 , channel) , name='input8x')

        mod2 = self.get_RDN_pass(depth = 20 , scale = 2)
        mod2.name = 'modelx_2x'
        zoom2x = mod2(inp)
        loss2x = self.VGG_content_loss( x= zoom2x , inp = inp2x , scale = 2)
        self.discriminatorx_2x = self.Adversarial_Loss(scale = 2)
        self.discriminatorx_2x.name = 'discriminatorx_2x_model'
        self.discriminatorx_2x.compile(loss = 'binary_crossentropy' , optimizer = Adam(lr = lr / 10),metrics=['accuracy'])
        self.make_trainable(self.discriminatorx_2x , False )
        gan_loss = self.discriminatorx_2x(zoom2x)
        self.modelx_2x = Model(inputs = [inp,inp2x] , outputs=[loss2x , zoom2x , gan_loss])
        self.modelx_2x.compile(loss = ['mse' , self.L1_loss , 'binary_crossentropy'], loss_weights=[self.content_lambda , self.mse_lambda , self.adv_lambda] , optimizer = self.adam , metrics={'discriminatorx_2x_model':'accuracy'})
        

        ###################################################
        mod4 = self.get_RDN_pass(depth = 20 , scale = 4)
        mod4.name = 'model2x_4x'
        zoom4x = mod4(zoom2x)
        loss4x = self.VGG_content_loss(x = zoom4x , inp = inp4x , scale = 4)
        
        zoom4x_inp2x = mod4(inp2x)
        loss4x_inp2x = self.VGG_content_loss(x = zoom4x_inp2x , inp = inp4x , scale = 4)
        self.discriminator2x_4x = self.Adversarial_Loss(scale = 4)
        self.discriminator2x_4x.name = 'discriminator2x_4x_model'
        self.make_trainable(self.discriminatorx_2x , False )
        gan_loss = self.discriminator2x_4x(zoom4x_inp2x)
        self.model2x_4x = Model(inputs = [inp2x,inp4x] , outputs = [loss4x_inp2x , zoom4x_inp2x , gan_loss])
        ####################################################


        ####################################################
        mod8 = self.get_RDN_pass(depth = 20 , scale = 8)
        mod8.name = 'model4x_8x'
        zoom8x = mod8(zoom4x)
        loss8x = self.VGG_content_loss(x = zoom8x , inp = inp8x , scale = 8)

        zoom8x_inp4x = mod8(inp4x)
        loss8x_inp4x = self.VGG_content_loss(x = zoom8x_inp4x , inp = inp8x , scale = 8)        
        self.discriminator4x_8x = self.Adversarial_Loss(scale = 8)
        self.discriminator4x_8x.name = 'discriminator4x_8x_model'
        gan_loss = self.discriminator4x_8x(zoom8x_inp4x)
        self.model4x_8x = Model(inputs = [inp4x,inp8x] , outputs = [loss8x_inp4x , zoom8x_inp4x , gan_loss])
        ####################################################

        model = Model(inputs= inp , outputs=[zoom2x , zoom4x , zoom8x])

        if chk >=0 :
            print("loading existing weights !!!")
            model.load_weights('model_iter'+str(chk)+'.h5')
        
        self.inference_model = model

        #model = Model(inputs=[inp , inp2x , inp4x , inp8x] , outputs = [loss2x , loss4x , loss8x , zoom2x , zoom4x , zoom8x])

       
        
        # # multi gpu setting

        # if gpu < 0:
        print("Running in multi gpu mode expecting 3 gpus , check source code")
        #self.multi_gpu_model = multi_gpu_model(model, gpus=3)
        # self.make_trainable(self.discriminatorx_2x , False )
        self.multi_gpu_modelx_2x = multi_gpu_model(self.modelx_2x , gpus=3)
        # self.make_trainable(self.discriminatorx_2x , True )
        self.multi_gpu_discriminatorx_2x = multi_gpu_model(self.discriminatorx_2x , gpus=3)

        self.multi_gpu_model2x_4x = multi_gpu_model(self.model2x_4x , gpus=3)
        self.multi_gpu_discriminator2x_4x = multi_gpu_model(self.discriminator2x_4x , gpus = 3)
        
        self.multi_gpu_model4x_8x = multi_gpu_model(self.model4x_8x , gpus=3)
        self.multi_gpu_discriminator4x_8x = multi_gpu_model(self.discriminator4x_8x , gpus = 3)
            
        print("Using Layers for content loss ,  " , self.content_loss_layers)
        ## Entire Model compilation
        #self.multi_gpu_model.compile(loss=['mse','mse','mse',self.L1_loss,self.L1_loss,self.L1_loss], optimizer=self.adam)
        
        # self.make_trainable(self.discriminatorx_2x , True )
        self.multi_gpu_discriminatorx_2x.compile(loss = 'binary_crossentropy' , optimizer = Adam(lr = lr / 10),metrics=['accuracy'])
        # self.make_trainable(self.discriminatorx_2x , False )
        self.multi_gpu_modelx_2x.compile(loss = ['mse' , self.L1_loss , 'binary_crossentropy'], loss_weights=[self.content_lambda , self.mse_lambda , self.adv_lambda] , optimizer = self.adam , metrics={'discriminatorx_2x_model':'accuracy'})

        self.multi_gpu_discriminator2x_4x.compile(loss = 'binary_crossentropy' , optimizer = Adam(lr = lr / 10),metrics=['accuracy'])
        self.make_trainable(self.discriminator2x_4x , False )
        self.multi_gpu_model2x_4x.compile(loss = ['mse' , self.L1_loss , 'binary_crossentropy'], loss_weights=[self.content_lambda , self.mse_lambda , self.adv_lambda] , optimizer = self.adam , metrics={'discriminator2x_4x_model':'accuracy'})

        self.multi_gpu_discriminator4x_8x.compile(loss = 'binary_crossentropy' , optimizer = Adam(lr = lr / 10),metrics=['accuracy'])
        self.make_trainable(self.discriminator4x_8x , False )
        self.multi_gpu_model4x_8x.compile(loss = ['mse' , self.L1_loss , 'binary_crossentropy'], loss_weights=[self.content_lambda , self.mse_lambda , self.adv_lambda] , optimizer = self.adam , metrics={'discriminator4x_8x_model':'accuracy'})

        # self.multi_gpu_model4x_8x.compile(loss = ['mse' , self.L1_loss] , optimizer = adam)
        # self.multi_gpu_model2x_4x.compile(loss = ['mse' , self.L1_loss] , optimizer = adam)
        if visualize == True:
            self.modelx_2x.summary()
            self.discriminatorx_2x.summary()

            self.model2x_4x.summary()
            self.discriminator2x_4x.summary()
            
            self.model4x_8x.summary()
            self.discriminator4x_8x.summary()
            
            self.visualize()
            self.get_model().summary()
        
    def make_trainable(self , net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val

    def fit_discriminator(self , x , x2 , x4 , x8 , batch_size = 16 , epoch = 5):
        
        x = x/255
        x2 = x2/255
        x4 = x4/255
        x8 = x8/255

        disc_y = np.zeros(x.shape[0]*2)
        disc_y[:x.shape[0]] = 1
        
        print(" Training Discriminator for x -> 2x generation model")
        x2_gen = self.multi_gpu_modelx_2x.predict([x,x2] , verbose = 1)[1]
        disc_x = np.concatenate([x2 , x2_gen])
        # s = (disc_x.shape[0] % batch_size)
        j = 0
        dic_acc = 0.0
        while dic_acc < 0.95:
            print("2x Discriminator epoch ",j)
            dic_acc =  self.multi_gpu_discriminatorx_2x.fit( x = disc_x[:] , y = disc_y[:] , batch_size = batch_size , epochs = 1).history['acc'][0] 
            print("2x Avg accuracy :",dic_acc)
            j += 1
            if j > epochs:
                break
        

        # print(" Training Discriminator for 2x -> 4x generation model")
        # x4_gen = self.multi_gpu_model2x_4x.predict([x2,x4] , verbose = 1)[1]
        # disc_x = np.concatenate([x4 , x4_gen])
        # s = (disc_x.shape[0] % batch_size)
        # j = 0
        # dic_acc = 0.0
        # while dic_acc < 0.95:
        #     print("4x Discriminator epoch ",j)
        #     dic_acc =  self.multi_gpu_discriminator2x_4x.fit( x = disc_x[:-s] , y = disc_y[:-s] , batch_size = batch_size , epochs = 1).history['acc'][0] 
        #     print("4x Avg accuracy :",dic_acc)
        #     j += 1
        #     if j > epochs:
        #         break
        
        # print(" Training Discriminator for 4x -> 8x generation model")
        # x8_gen = self.multi_gpu_model4x_8x.predict([x4,x8] , verbose = 1)[1]
        # disc_x = np.concatenate([x8 , x8])
        # ## Its a hack for multi gpu training may or may nor work.....
        # s = (disc_x.shape[0] % batch_size)
        # j = 0
        # dic_acc = 0.0
        # while dic_acc < 0.95:
        #     print("8x Discriminator epoch ",j)
        #     dic_acc = self.multi_gpu_discriminator4x_8x.fit( x = disc_x[:-s] , y = disc_y[:-s] , batch_size = batch_size , epochs = 1).history['acc'][0]
        #     print("8x Avg accuracy :",dic_acc)
        #     j += 1
        #     if j > epochs:
        #         break


    def fit(self , x , x2 , x4 , x8 ,batch_size=16 , epoch = 50 ):
        # with tf.device('/gpu:'+str(gpu)):
        x = x/255
        x2 = x2/255
        x4 = x4/255
        x8 = x8/255


        zeros = np.zeros(x.shape[0])
        gan_y = np.ones(x.shape[0])
        
        print("Training GAN + Perecptual loss for x -> 2x generative model")
        j = 0
        gen_acc = 0.0
        while gen_acc <= 0.5:
            print("Generator epoch ",j)
            gen_acc = self.multi_gpu_modelx_2x.fit(x = [x,x2] , y = [zeros , x2 , gan_y] ,batch_size = batch_size  , verbose =1 , nb_epoch = 1).history['discriminatorx_2x_model_acc'][0]
            print("2x Avg accuracy :",gen_acc)
            j += 1
            if j > epochs:
                break
        
        
        
        # print("Training GAN + Perceptual Loss for 2x -> 4x generative model")
        # x2_gen = self.multi_gpu_modelx_2x.predict([x,x2] , verbose = 1)[1]
        # j = 0
        # gen_acc = 0.0
        # while gen_acc < 0.8:
        #     print("Generator epoch ",j)
        #     gen_acc = self.multi_gpu_model2x_4x.fit(x = [x2_gen,x4] , y = [zeros , x4 , gan_y] ,batch_size = batch_size , verbose =1 , nb_epoch = 1).history['discriminator2x_4x_model_acc'][0]
        #     print("4x Avg accuracy :",gen_acc)
        #     j += 1
        #     if j > epochs:
        #         break
        
        
        # print("Training GAN + Perceptual Loss for 2x -> 4x generative model")
        # x4_gen = self.multi_gpu_model2x_4x.predict([x2,x4] , verbose = 1)[1]
        # j = 0
        # gen_acc = 0.0
        # while gen_acc < 0.8:
        #     print("Generator epoch ",j)
        #     s = (x8.shape[0] % (batch_size//3))
        #     gen_acc = self.multi_gpu_model4x_8x.fit(x = [x4_gen[:-s],x8[:-s]] , y = [zeros[:-s] , x8[:-s] , gan_y[:-s]] ,batch_size = batch_size//3 , verbose =1 , nb_epoch = 1).history['discriminator4x_8x_model_acc'][0]
        #     print("8x Avg accuracy :",gen_acc)
        #     j += 1
        #     if j > epochs:
        #         break
    def predict(self , p):
        p = p/255
        g = self.get_model().predict(np.array(p))
        g[0] = np.array(g[0]*255, np.uint8)
        g[1] = np.array(g[1]*255, np.uint8)
        g[2] = np.array(g[2]*255, np.uint8)
        return g
        

if __name__ == '__main__':
    CHANNEL = 3
    out_patch_size =  128
    inp_patch_size = 16
    DATA = DATA(folder = folder , patch_size = out_patch_size)
    if not test_only:
        DATA.load_data(folder=folder)
        x = DATA.training_patches_8x
        x2 = DATA.training_patches_4x
        x4 = DATA.training_patches_2x
        x8 = DATA.training_patches_Y

    net = SRResnet(lr = learning_rate , chk = chk  , test= test_only , content_lambda = content_lambda , mse_lambda = mse_lambda , adv_lambda=adv_lambda , patch_size=inp_patch_size ,visualize = not test_only)
        

    image_name = values.test_image

    try:
        img = cv2.imread(image_name)
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        print("Bad image path check the name or path !!")
        exit()

    if not zoom:
        R = DATA.patch_size - img.shape[0] % DATA.patch_size
        C = DATA.patch_size - img.shape[1] % DATA.patch_size

        img = np.pad(img, [(0,R),(0,C),(0,0)] , 'constant')

        Image.fromarray(img).save("test_image_padded.png")

        lr_2x_img = cv2.resize(img , (int(img.shape[1]/2),int(img.shape[0]/2)) ,cv2.INTER_CUBIC)
        Image.fromarray(lr_2x_img).save("test_2x_lr_padded.png")

        lr_4x_img = cv2.resize(img , (int(img.shape[1]/4),int(img.shape[0]/4)) ,cv2.INTER_CUBIC)
        Image.fromarray(lr_4x_img).save("test_4x_lr_padded.png")

        lr_8x_img = cv2.resize(img , (int(img.shape[1]/8),int(img.shape[0]/8)) ,cv2.INTER_CUBIC)
        Image.fromarray(lr_8x_img).save("test_8x_lr_padded.png")                

        p , r , c = DATA.patchify(lr_8x_img,scale=8) 

        if not os.path.isdir('Results'):
            os.mkdir('Results') 

    if not test_only:
        
        for i in range(chk+1,tryout):
            gen_acc = 0.0
            dic_acc = 0.0
            print("tryout no: ",i)
            
            # samplev = np.random.random_integers(0 , x.shape[0]-1 , sample)
            
            if adv_lambda > 0.0:
                net.fit_discriminator(x, x2 , x4 , x8 , batch_size , epoch = epochs)
            
            net.fit(x , x2 , x4, x8 , batch_size , epoch = epochs)
            
            net.get_model().save_weights('model_iter'+str(i)+'.h5')
            g = net.predict(np.array(p))
            gen2x = DATA.reconstruct(g[0] , r , c , scale=4)
            gen4x = DATA.reconstruct(g[1] , r , c , scale=2)
            gen8x = DATA.reconstruct(g[2] , r , c , scale=1)
            d = 'Results/'+str(i)
            if not os.path.isdir(d):
                os.mkdir(d)
            Image.fromarray(gen2x).save(d+"/test_2x_gen_.png")
            Image.fromarray(gen4x).save(d+"/test_4x_gen_.png")
            Image.fromarray(gen8x).save(d+"/test_8x_gen_.png")
            print("Reconstruction Gain:", PSNRLossnp(img , gen8x))
    else :
        if zoom:
            gz , r2 , c2 = DATA.patchify(img , scale = 8)
            gz = net.get_model().predict(np.array(gz))[2]       
            genz = DATA.reconstruct(gz , r2 , c2 , scale = 1)
            Image.fromarray(genz).save("test_image_zoomed_8x.png")      
        else:
            g = net.get_model().predict(np.array(p))[2]
            gen = DATA.reconstruct(g , r , c , scale=8)     
            Image.fromarray(gen).save("test_8x_gen_.png")
            print("Reconstruction Gain:", PSNRLossnp(img , gen))