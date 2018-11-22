
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras import backend as K

from sklearn.metrics import f1_score as off1

class F1(object):

    def __init__(self):
        self.threshold = 0.01
        self.final_t = 0.0

    def f1(self,y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return:
        """
        y_pred = tf.cast(tf.greater(tf.clip_by_value(y_pred, 0, 1), self.threshold), dtype=tf.float32)
        tp = tf.reduce_sum(tf.cast(y_true * y_pred, dtype=tf.float32), axis=0)
        tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), dtype=tf.float32), axis=0)
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, dtype=tf.float32), axis=0)
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), dtype=tf.float32), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2 * p * r / (p + r + K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return tf.reduce_mean(f1)


    def f1_full_validation(self,y_true,y_pred,class_num,save_path=None):
        """


        :param y_true:
        :param y_pred:
        :param class_num:
        :param save_path:
        """
        rng = np.arange(0, 1, 0.001)
        f1s = np.zeros((rng.shape[0], class_num))

        # reshape, if problem is mutli-label problem
        y_true = np.reshape(y_true,(y_true.shape[0],-1,class_num))
        y_pred = np.reshape(y_pred,(y_pred.shape[0],-1,class_num))

        for j,t in enumerate(tqdm(rng,'Updating the F1 threshold')):
            for i in range(class_num):
                p = np.array(y_pred[...,i]>t,dtype=np.int8)
                scoref1 = off1(y_true[...,i].reshape(-1),p.reshape(-1),average='binary')
                f1s[j,i] = scoref1

        print('Individual F1-scores for each class:')
        print(np.max(f1s, axis=0))
        print('Macro F1-score CV =', np.mean(np.max(f1s, axis=0)))

        T = np.empty(class_num*y_true.shape[1])

        F_T = np.empty(class_num)

        # Duplicate the threshold j times
        for j in range(y_true.shape[1]):
            for i in range(class_num):
                T[i+j*class_num] = rng[np.where(f1s[:, i] == np.max(f1s[:, i]))[0][0]]
                F_T[i] = rng[np.where(f1s[:, i] == np.max(f1s[:, i]))[0][0]]
        print('Probability threshold maximizing CV F1-score for each class:')
        print(F_T)

        self.threshold = T
        self.final_t = F_T



