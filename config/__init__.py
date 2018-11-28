from .common_cfg import *


import tensorflow as tf


flags = tf.app.flags

FLAGS = flags.FLAGS


if FLAGS.model_name == "faster-rcnn":
    from .faster_rcnn_cfg import *
