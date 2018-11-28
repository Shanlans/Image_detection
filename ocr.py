from keras.optimizers import *

from data import *
from models import *
from task import *

epochs = 100


def main(argv):
    aug_seq = seq

    data_path = "./metadata/OCR/chezai_20181121"

    ocr_train_data = OCRDataGen(data_path,
                                characterset='BCEGNS1247',
                                fixed_length=2,
                                batch_size=5,
                                resize_shape=(600, 1000),
                                shuffle=True,
                                augment=True,
                                aug_seq=aug_seq)
    ocr_val_data = OCRDataGen(data_path + '_val',
                              characterset='BCEGNS1247',
                              fixed_length=2,
                              batch_size=5,
                              resize_shape=(600, 1000),
                              shuffle=False,
                              augment=False)

    ocr_test_data = OCRDataGen(data_path + '_test',
                               characterset='BCEGNS1247',
                               fixed_length=2,
                               batch_size=5,
                               resize_shape=(600, 1000),
                               shuffle=False,
                               augment=False)

    model = VGG(input_shape=[600, 1987, 3])()
    layer_shape = [1000, 1000]
    activation = ['relu', 'relu']
    model = multi_label_classifier(base_model=model,
                                   class_num=ocr_train_data.class_num,
                                   label_max_length=2,
                                   layer_shape=layer_shape,
                                   activation=activation)
    model.summary()

    adam = Adam(lr=1e-5)
    #
    # fscore = F1()
    #
    # f1 = fscore.f1
    # metrics = ['acc',f1]
    # model.compile(optimizer=adam,
    #               loss='categorical_crossentropy',
    #               metrics=metrics)
    #
    # callbacks = callback_build(val_data_generate=ocr_test_data,
    #                            batch_size=32,
    #                            log_dir='./logs',
    #                            tb_mode='batch',
    #                            tb_period=20,
    #                            save_ckpt_period=1,
    #                            ckpt_monitor='val_acc',
    #                            lr_monitor='val_loss',
    #                            clear_log=True,
    #                            threshold_update_fn=fscore.f1_full_validation)
    #
    # hist = model.fit_generator(
    #     ocr_train_data,
    #     steps_per_epoch=len(ocr_train_data),
    #     epochs=epochs,
    #     validation_data=ocr_val_data,
    #     validation_steps=len(ocr_train_data),
    #     callbacks=callbacks)
    #
    # print('User should use threshold like this {}'.format(fscore.final_t))
    #
    # draw_hist(hist=hist,metrics=metrics,clear_log=True)
    # draw_hist(hist=hist,metrics=metrics,clear_log=True)


if __name__ == "__main__":
    # set_seed(666)
    tf.app.run()
