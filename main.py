
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
epochs = 10


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
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc', f1])

    callbacks = callback_build(val_data_generate=cifa_val_data,batch_size=32,log_dir='./logs',tb_mode='batch',tb_period=20,save_ckpt_period=1,ckpt_monitor='val_acc',lr_monitor='val_loss',clear_log=True)

    model.fit_generator(
        cifa_train_data,
        steps_per_epoch=len(cifa_train_data),
        epochs=epochs,
        validation_data=cifa_val_data,
        validation_steps=8,
        callbacks=callbacks)



if __name__ == "__main__":


    #set_seed(666)
    tf.app.run()