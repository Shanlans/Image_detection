
import keras.backend as K

def mape_acc(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 1-(100. * K.mean(diff, axis=-1))