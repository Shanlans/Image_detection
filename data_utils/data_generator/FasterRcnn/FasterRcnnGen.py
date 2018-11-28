import math
import cv2
import numpy as np
import imgaug as ia

import tensorflow as tf
from keras.utils import np_utils, Sequence

from data_utils.data_parser import get_pascal_detection_data

from .Rpn_utils import img_resize, calc_rpn
from .get_feature_map_shape import *

from ..data_augmentor import faster_rcnn_aug

flags = tf.app.flags

FLAGS = flags.FLAGS

_FEATURE_MAP_SHAPE = {
    'VGG16': vgg_16_output_shape,
}


class FasterRcnnDataGen(Sequence):

    def __init__(self, data_dir, min_resize_shape=600, front_end='VGG16', train_phase=True, shuffle=False,
                 use_cache=False,aug=True):
        """

        :param data_dir:
        :param min_resize_shape: An Integer. Default is 600. In the paper, author resize the image shorter size to 600 pixels
        :param front_end: A String. Default is Vgg-16. Which front-end to use. To help getting the output feature map shape
        :param train_phase: Boolean, Default is 'True' which will return the training data (images and labels),
                            otherwise return the validation data (images and labels)
        :param shuffle: Boolean. Default is 'False' which would not shuffle data set, otherwise shuffle it
        :param use_cache: Boolean, Default is 'True' which cache the every image when the first epochs, second epochs,
                            no need to reload image cause it will waste the time do I/O operation
        :param aug: Boolean. Default is True. Open/off Augmentation
        """
        self.data_dir = data_dir
        self.batch_size = 1 #faster-RCNN batch size should be equal to 1, cause the rpn output shape is different from each other image. But I think it can be optimized.
        self.min_resize_shape = min_resize_shape
        self.front_end = front_end
        self.train_phase = train_phase
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.aug = aug

        all_images, classes_count, class_mapping = get_pascal_detection_data(self.data_dir)

        self.class_name = class_mapping.keys()

        self.class_num = len(classes_count)

        print('Total image number {}'.format(len(all_images)))

        if self.train_phase:
            # self.images content:
            # {'filepath','width',''
            self.images = np.array([s for s in all_images if s['imageset'] == 'trainval'])
            print('Num train samples {}'.format(len(self.images)))
        else:
            self.images = np.array([s for s in all_images if s['imageset'] == 'test'])
            print('Num val samples {}'.format(len(self.images)))

        self.on_epoch_end()

    def __len__(self):
        """
        Number of batch in the Sequence.
        Returns:
            The number of batches in the Sequence.
        """

        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        """
        Gets batch at position `index`.
        Arguments:
            index: position of the batch in the Sequence.
        Returns:
            A batch
        """

        # Train RPN only support batch size = 1, because the output shape is different
        index = self.indexes[idx]

        image_batch = self.images[index]

        image_batch, rpn_batch,original_image_batch = self.__load_data(image_batch)

        return image_batch,rpn_batch

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __load_data(self,image):


        img_data_aug, x_img = faster_rcnn_aug(image, FLAGS, augment=self.aug)

        # get image dimensions for resizing,resize the image so that smalles side is length = 600px
        x_img, original_size, new_size = img_resize(x_img, self.min_resize_shape)



        y_rpn_cls, y_rpn_regr = calc_rpn(img_data=img_data_aug, original_size=original_size, new_size=new_size,
                                         img_length_calc_function=_FEATURE_MAP_SHAPE[self.front_end])



        return x_img, [y_rpn_cls,y_rpn_regr],img_data_aug

