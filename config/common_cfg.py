import tensorflow as tf


flags = tf.app.flags



flags.DEFINE_integer('batch_size',3,'Training batch size')

flags.DEFINE_enum('job_type','Segmentation',['Classification','Segmentation','Detection'],'Choose one from "Classification","Segmentation","Detection"')

flags.DEFINE_enum('model_name','auto-encoder',['faster-rcnn','auto-encoder'],'Choose one from "faster-rcnn",...')

flags.DEFINE_enum('front_end','VGG16',['VGG16','VGG19',],'Choose one from "VGG16","VGG19"...')

flags.DEFINE_float('base_learning_rate',1e-6,'Start learning rate')

flags.DEFINE_integer('epochs',100,'Training epochs')