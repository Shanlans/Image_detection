
import os
import shutil

from keras.optimizers import *
from keras.callbacks import *

from data import *
from models import *
from task import *
from monitor import *
from train_utils import *

num_classes = 10
epochs = 1

model_dir = './logs/models'
log_dir = './logs/tensorboard'



def main(argv):

    aug_seq = seq
    cifa_train_data = CIFARDataGen(batch_size=32,resize_shape=(224,224),cifa_10=True,train_phase=True,shuffle=True,augment=False,aug_seq=aug_seq)
    cifa_val_data = CIFARDataGen(batch_size=32,resize_shape=(224,224),cifa_10=True,train_phase=False,shuffle=True,augment=False)


    model = VGG(input_shape=[224, 224, 3])()
    layer_shape = [1000, 1000, num_classes]
    activation = ['relu', 'relu', 'softmax']
    model = classifier(model, layer_shape=layer_shape, activation=activation)
    model.summary()


    adam = Adam(lr=1e-6)
    adam = Adam(lr=1e-6)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy','categorical_accuracy', f1])


    callback = []
    reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')
    checkpoint_acc = ModelCheckpoint(model_dir+'/model', monitor='val_categorical_accuracy', verbose=1,
                                     save_best_only=True, save_weights_only=False, mode='max', period=1)
    tensorboard = CustomTensorboard(log_dir=log_dir,histogram_freq=0,write_images=True,write_grads=True,write_graph=True,update_freq='batch')

    callback.append(reduceLROnPlato)
    callback.append(checkpoint_acc)
    callback.append(tensorboard)


    model.fit_generator(
        cifa_train_data,
        steps_per_epoch=len(cifa_train_data),
        epochs=epochs,
        validation_data=cifa_val_data,
        validation_steps=8,
        callbacks=callback)



if __name__ == "__main__":

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    tf.app.run()