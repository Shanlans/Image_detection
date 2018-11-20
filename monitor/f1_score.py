import tensorflow as tf

from keras import backend as K

THRESHOLD = 0.05
def f1(y_true, y_pred):
    y_pred = tf.cast(tf.greater(tf.clip_by_value(y_pred, 0, 1), THRESHOLD), dtype=tf.float32)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, dtype=tf.float32), axis=0)
    tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), dtype=tf.float32), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, dtype=tf.float32), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), dtype=tf.float32), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return tf.reduce_mean(f1)