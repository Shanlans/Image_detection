
import numpy as np
import tensorflow as tf
import imgaug as ia



def set_seed(seed=666):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    ia.seed(seed)
