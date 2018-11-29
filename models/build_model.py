
import tensorflow as tf

from keras.layers import Input
from keras.models import Model

from .frontend import VGG
from .faster_rcnn import *
from .AutoEncoder import *

flags = tf.app.flags

FLAGS = flags.FLAGS



_SUPPORTED_MODELS = []

_SUPPORTED_FRONTENDS = ["Xception",
                        "VGG16",
                        "VGG19",
                        "ResNet50",
                        "InceptionV3",
                        ]



class BuildModel(object):

    def __init__(self,front_end,model_name,job_type,input_shape=None,initial_weights=None):
        """

        :param initial_weights:
        :param front_end:
        :param model_name:
        :param input_shape:
        :param job_type:
        :param weights: String, Defaule is None. If None, initial weights from pretrain model, otherwise initial from string path
        """
        self.front_end = front_end
        self.model_name = model_name
        self.input_shape = input_shape
        self.job_type = job_type





    def build_model(self):
        if self.job_type == 'Classification':
            self.inputs = Input(shape=self.input_shape, dtype='float32', name='inputs')
            frontend_model = self.build_frontend()
            frontend_model.summary()
            return frontend_model

        elif self.job_type == 'Detection':
            self.inputs = Input(shape=(None,None,3),dtype='float32',name='inputs')
            frontend_model = self.build_frontend()
            if self.model_name == 'faster-rcnn':
                self.num_anchors = len(FLAGS.anchor_box_scales) * len(eval(FLAGS.anchor_box_ratios))
                rpn = RPN(frontend_model.output, self.num_anchors)
                rpnModel = Model(self.inputs,rpn,name='RPN')
                rpnModel.summary()
                return rpnModel

        elif self.job_type == 'Segmentation':
            self.inputs = Input(shape=(None,None,3),dtype='float32',name='inputs')
            encoder_model = self.build_frontend()
            if self.model_name == 'auto-encoder':
                decode = vgg16_decoder(encoder_model.output)
                autoEncodermodel = Model(self.inputs,decode,name='AutoEncoder')

                autoEncodermodel.summary()
                return autoEncodermodel


    def build_frontend(self):
        if self.front_end not in _SUPPORTED_FRONTENDS:
            raise ValueError(
                "The frontend you selected is not supported. The following models are currently supported: {0}".format(_SUPPORTED_FRONTENDS))


        if self.front_end == 'VGG16':
            features,end_point = VGG(vgg_version='vgg16',input_tensor=self.inputs)()
            end_point = features.get_layer(end_point).output

        elif self.front_end == 'VGG19':
            features, end_point = VGG(vgg_version='vgg19',input_tensor=self.inputs)()
            end_point = features.get_layer(end_point).output
        else:
            pass

        if self.job_type == 'Classification':
            return features
        elif self.job_type == 'Detection':
            return Model(inputs=features.input, outputs=end_point)
        elif self.job_type == 'Segmentation':
            return Model(inputs=features.input, outputs=end_point)


