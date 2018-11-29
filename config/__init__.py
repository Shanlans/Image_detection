from .common_cfg import *


import tensorflow as tf


flags = tf.app.flags

FLAGS = flags.FLAGS


if FLAGS.model_name == "faster-rcnn":
    from .faster_rcnn_cfg import *
elif FLAGS.model_name == 'auto-encoder':
    from .auto_encoder_cfg import *
