

import tensorflow as tf
from keras.layers import *




def vgg16_decoder(x):

    with tf.variable_scope('vgg16-decoder'):
        # Block 1
        x = Conv2D(filters=512,
                   kernel_size=(3,3),
                   activation='relu',
                   padding='same',
                   name='deblock1_conv1')(x)
        x = Conv2D(filters=512,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='deblock1_conv2')(x)
        x = Conv2D(filters=512,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='deblock1_conv3')(x)

        # Block2
        x = UpSampling2D(size=(2,2),name='block2_up')(x)
        x = Conv2D(filters=512,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='deblock2_conv1')(x)
        x = Conv2D(filters=512,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='deblock2_conv2')(x)
        x = Conv2D(filters=512,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='deblock2_conv3')(x)

        # Block3
        x = UpSampling2D(size=(2, 2), name='block3_up')(x)
        x = Conv2D(filters=256,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='deblock3_conv1')(x)
        x = Conv2D(filters=256,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='deblock3_conv2')(x)
        x = Conv2D(filters=256,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='deblock3_conv3')(x)

        # Block4
        x = UpSampling2D(size=(2, 2), name='block4_up')(x)
        x = Conv2D(filters=128,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='deblock4_conv1')(x)
        x = Conv2D(filters=128,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='deblock4_conv2')(x)

        # Block5
        x = UpSampling2D(size=(2, 2), name='block5_up')(x)
        x = Conv2D(filters=64,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='deblock5_conv1')(x)
        x = Conv2D(filters=64,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name='deblock5_conv2')(x)

        # Output Block
        outputs = Conv2D(filters=3,
                        kernel_size=(3,3),
                        activation='sigmoid',
                        padding='same',
                        name='outputs')(x)

    return outputs


