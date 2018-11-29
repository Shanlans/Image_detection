
import tensorflow as tf

from keras.layers import *

def rpn(base_layers, num_anchors):
    with tf.variable_scope('RPN') as scope:
        # 滑窗 3*3 生成 256 维，因为是 每个点滑动，每个点对应的都有9个anchor 所以要padding = same
        x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
            base_layers)

        # 1*1 相当于 fcn
        x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
        x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr]