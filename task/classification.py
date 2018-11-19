
from keras.models import Model
from keras.layers import *


def classifier(base_model,layer_shape=[],activation=[]):
    '''
    :param base_model: base model from feature extraction
    :param class_num: integer, how many class, will be used in the last dense layer
    :param class_shape: list, classifier will contain several fc layers
    :param activation: list, each fc layer can use different activation function,usually the last activation function is 'softmax'
    :return: model to fit
    '''

    if len(layer_shape) != len(activation):
        assert 'last layers number [{}] should have the same number of activation [{}]'.format(layer_shape,activation)

    if activation[-1] != 'softmax':
        warnings.warn('The last layers activation function usually is "softmax", now is {}'.format(activation[-1]))

    x = base_model.output

    for i in range(len(layer_shape)):
        x = Dense(units=layer_shape[i],
                  activation=activation[i],name='fc_%d'%i)(x)


    predictions = x

    model = Model(inputs=base_model.input,outputs=predictions)

    return model


