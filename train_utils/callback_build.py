import os
import shutil


from keras.callbacks import *
from keras.utils import Sequence


from .callback import *


def callback_build(val_data_generate,batch_size,log_dir='./logs', tb_mode='batch', tb_period=20, save_ckpt_period=1,
                   ckpt_monitor='val_acc', lr_monitor='val_loss', clear_log=True,threshold_update_fn=None):
    """

    # General

    :param val_data_generate: An Sequence (keras.utils.Sequence) Object. To give the data Sequence obj to tensorboard for summary purpose
    :param log_dir: String. Default to './logs', parents directory for tensorboard and model saving
    :param batch_size: Integer. Batch size
    :param clear_log: Boolean. Default to True, if train startover, need to clean the logs file

    # Tensorboard
    :param tb_mode: String. Default to 'batch', enum from ['batch','epoch'].
                    If choose 'batch' will summary the training situation at each batch end,
                    otherwise will summary it at each epoch end
    :param tb_period: Integer. Default to 20. How long will save tensorboard.
                    If 'batch' & tb_period, e.g. tb_period =20, will save tensorboard every 20 batches.

    # Checkpoint
    :param save_ckpt_period: Integer. Default to 1, how many epoches will save checkpoint
    :param ckpt_monitor: String. Default to 'val_acc', setting it to monitoring the value you want to check, usually is 'acc'.

    # Learning rate
    :param lr_monitor: String. Default to 'val_loss', if lr_monitor's value is not change, lr_scheduler will change the learning rate

    :return: List. A list container contains every callback obj

    # Full validation
    :param threshold_update_fn: An function to calculate threshold of each class, and update for next f1 calculate
    """




    # Checkpoint
    if not isinstance(val_data_generate, Sequence):
        raise ValueError("val_data_generate must be provided, and must be a generator (like Sequence obj).")

    ckpt_name = "weights-improvement-{epoch:02d}-{%s:.2f}.hdf5" % ckpt_monitor
    model_dir = os.path.join(log_dir,'models', ckpt_name)

    if 'acc' in ckpt_monitor:
        save_mode = 'max'
    elif 'loss' in ckpt_monitor:
        save_mode = 'min'

    checkpoint_save = ModelCheckpoint(model_dir, monitor=ckpt_monitor, verbose=1,
                                      save_best_only=True, save_weights_only=False, mode=save_mode,
                                      period=save_ckpt_period)


    # Learning rate
    lr_scheduler = ReduceLROnPlateau(monitor=lr_monitor, factor=0.5, patience=3, verbose=1, mode='min')

    # Tensorboard
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    tensorboard = CustomTensorboard(log_dir=tensorboard_dir,
                                    histogram_freq=tb_period,
                                    validation_data=val_data_generate,
                                    batch_size=batch_size,
                                    write_images=False,
                                    write_grads=False,
                                    write_graph=True,
                                    update_freq=tb_mode)

    # FullValidation
    validation_result_dir = os.path.join(log_dir,'val')
    fullVal = FullEvaluateForMultiLabels(validation_data=val_data_generate,batch_size=batch_size,log_dir=validation_result_dir,threshold_update_fn=threshold_update_fn)

    if clear_log:
        if os.path.isdir(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
        if os.path.isdir(validation_result_dir):
            shutil.rmtree(validation_result_dir)
        if os.path.isdir(os.path.join(log_dir, 'models')):
            shutil.rmtree(os.path.join(log_dir, 'models'))


    if not os.path.isdir(os.path.join(log_dir,'models')):
        os.makedirs(os.path.join(log_dir,'models'))

    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    if not os.path.isdir(validation_result_dir):
        os.makedirs(validation_result_dir)

    return [checkpoint_save, lr_scheduler, tensorboard,fullVal]
