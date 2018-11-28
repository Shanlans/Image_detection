import tensorflow as tf


flags = tf.app.flags

flags.DEFINE_string('data_dir','./metadata/VOCdevkit','Where the data locate')

flags.DEFINE_integer('image_min_size',600,'Resized image shorted boundary length')


flags.DEFINE_multi_integer('anchor_box_scales',[128,256,512],'anchor box sizes')

flags.DEFINE_string('anchor_box_ratios','[[1,1],[1,2],[2,1]]','anchor box ratio [1:1,1:2,2:1]')

flags.DEFINE_integer('rpn_stride',16,'stride at the RPN (this depends on the network configuration)')

flags.DEFINE_float('rpn_max_overlap',0.7,'If rpn need over 70% area IOU')
flags.DEFINE_float('rpn_min_overlap',0.3,'If not rpn,IOU under 30% area')

## Image Augmentation

flags.DEFINE_boolean('use_horizontal_flips',True,'Image horizontal flip randomly')
flags.DEFINE_boolean('use_vertical_flips',True,'Image vertical flip randomly')
flags.DEFINE_boolean('rot_90',True,'Image rotate 90/180/270/0')

