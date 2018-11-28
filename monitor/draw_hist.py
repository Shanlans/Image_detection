import os
import shutil


import numpy as np
import matplotlib.pyplot as plt



def draw_hist(hist,metrics,save_path='./logs/statics/',clear_log=True):
    """
    Draw the hist figure after training finished
    :param hist: A `History` object. Its `History.history` attribute is
                a record of training loss values and metrics values
                 at successive epochs, as well as validation loss values
                and validation metrics values (if applicable).
    :param metrics: List. It should be same as model compile's metrics
    :param save_path: String. Default is './logs' Where to save the hist figure.
    :param clear_log: Boolean. Default is True. Clear the statics folder
    """

    if clear_log:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if not isinstance(metrics,list):
        raise ('Metrics should be a list obj, but now is {}'.format(type(metrics)))

    metrics.append('loss')

    metrics_num = len(metrics)

    for metric in metrics:
        fig = plt.figure(figsize=(5,5))

        if not isinstance(metric,str):
            metric = metric.__name__
        plt.plot(hist.epoch,hist.history[metric],label='Train-'+metric)
        plt.plot(hist.epoch,hist.history['val_'+metric],label='Validation-'+metric)
        plt.xlabel('epoch',fontsize=16.0)
        plt.legend()
        fig.suptitle(metric,fontsize=20.0)
        plt.savefig(save_path+metric+'.png')

