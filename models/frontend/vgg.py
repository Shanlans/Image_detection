
import os

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

import tensorflow as tf

class VGG(object):
    '''

    '''

    def __init__(self,vgg_version='vgg16',input_tensor=None,weights='imagenet',pooling='avg'):
        '''
        :rtype: object, VGG
        :param vgg_version: String type, Vgg version selection from 'vgg16' or 'vgg19'. Default is 'vgg16'
        :param input_tensor: A tensor, should 3 channel and channel_last format, height and weight should be no smaller than 32
        :param weights: String type,Initial network from official 'Imagenet' pretrain model,
                        or pass a weight dir path to load users trained weights
                        Default loading weights from official 'imagenet'
        :param pooling: String type, control the feature extractor with global 'avg' or 'max' pooling layer. Default is 'avg'
        '''

        self.vgg_version = vgg_version
        self.weights = weights
        self.pooling = pooling
        self.input_tensor = input_tensor

        if weights is 'imagenet':
            print('Initial {} from official pre-trained model "imagenet"'.format(self.vgg_version))
        elif weights is None:
            print('Baby sitting training start')
        elif os.path.splitext(weights)[-1] !='.h5':
            assert 'weights should bt H5 file, but now is {}'.format(os.path.splitext(weights)[-1])


        if vgg_version =='vgg16':
            self.base_model = self.__vgg16()
            self.endpoint = 'block5_conv3'
        elif vgg_version =='vgg19':
            self.base_model = self.__vgg19()
            self.endpoint = 'block5_conv4'
        else:
            assert('{} model is not supported now!!!'.format(self.vgg_version))


    def __vgg16(self):
        weights = self.weights
        pooling = self.pooling
        input_tensor = self.input_tensor
        base_model =VGG16(weights=weights,input_tensor=input_tensor,include_top=False,pooling=pooling)
        return base_model

    def __vgg19(self):
        weights = self.weights
        pooling = self.pooling
        input_tensor = self.input_tensor
        base_model = VGG19(weights=weights,input_tensor=input_tensor,include_top=False,pooling=pooling)
        return base_model

    def __call__(self, *args, **kwargs):
        return self.base_model,self.endpoint
