
from keras.models import Model
from keras.layers import *


def single_label_classifier(base_model,class_num,layer_shape=[],activation=[]):
    '''
    :param base_model: base model from feature extraction
    :param class_num: integer, how many class, will be used in the last dense layer
    :param class_shape: list, classifier will contain several fc layers
    :param activation: list, each fc layer can use different activation function,usually the last activation function is 'softmax'
    :return: model to fit
    '''

    if len(layer_shape) != len(activation):
        assert 'last layers number [{}] should have the same number of activation [{}]'.format(layer_shape,activation)

    x = base_model.output

    for i in range(len(layer_shape)):
        x = Dense(units=layer_shape[i],
                  activation=activation[i],name='fc_%d'%i)(x)

    x = Dense(units=class_num,activation='softmax',name='Classifier')(x)


    predictions = x

    model = Model(inputs=base_model.input,outputs=predictions)

    return model


def multi_label_classifier(base_model,class_num,label_max_length=None,layer_shape=[],activation=[]):
    '''
    :param base_model: base model from feature extraction
    :param class_num: integer, how many class, will be used in the last dense layes
    :param label_max_length: Interger. Default is None. If none will set it equal to class number,
                            otherwise, it must be set to maximum label length
    :param class_shape: list, classifier will contain several fc layers
    :param activation: list, each fc layer can use different activation function,usually the last activation function is 'softmax'
    :return: model to fit
    '''

    if len(layer_shape) != len(activation):
        assert 'last layers number [{}] should have the same number of activation [{}]'.format(layer_shape,activation)

    if label_max_length is None:
        label_max_length = class_num

    x = base_model.output

    for i in range(len(layer_shape)):
        x = Dense(units=layer_shape[i],
                  activation=activation[i],name='fc_%d'%i)(x)
    output_shape = class_num*label_max_length
    x = Dense(units=output_shape, activation='sigmoid', name='Classifier')(x)

    predictions = x

    model = Model(inputs=base_model.input,outputs=predictions)

    return model


