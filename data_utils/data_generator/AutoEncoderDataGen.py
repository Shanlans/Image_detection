import math
import cv2
import os
import copy
import numpy as np
import imgaug as ia

from keras.utils import np_utils, Sequence

from .data_augmentor import ae_aug,random_mask


class AutoEncoderDataGen(Sequence):

    def __init__(self, data_dir,batch_size, img_ext='bmp',resize_shape=None,train_phase=True, shuffle=True,
                 augment=True):
        """
        :param img_ext:
        :param data_dir:
        :param resize_shape:
        :param batch_size: An Integer, Batch size for training
        :param cifa_10: Boolean. Default is 'True' which means loading the cifa-10 dataset,
                        otherwise using the cifa-100 dataset
        :param train_phase: Boolean, Default is 'True' which will return the training data (images and labels),
                            otherwise return the validation data (images and labels)
        :param shuffle: Boolean. Default is 'False' which would not shuffle data set, otherwise shuffle it
        :param augment: Boolean. Default is 'False' which would not do augmentation for this batch of data
        :param aug_seq: ImgAug Sequential Object. Default is None. A simple sequence can follow this guide https://imgaug.readthedocs.io/en/latest/source/examples_basics.html#a-simple-and-common-augmentation-sequence
        """

        self.img_ext = img_ext
        self.resize_shape = resize_shape
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_phase = train_phase
        self.shuffle = shuffle
        self.augment = augment
        self.aug_seq = ae_aug


        images = []

        for root,folders,files in os.walk(self.data_dir):
            for file in files:
                if self.img_ext in os.path.splitext(file)[-1].lower():
                    images.append(os.path.join(root,file))
        self.images = np.array(images)
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
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]

        image_batch = self.images[indexes]

        image_batch = self.__load_image(image_batch)

        if self.resize_shape:
            # Resize images interpolation choice
            # ``nearest`` (identical to ``cv2.INTER_NEAREST``)
            # ``linear`` (identical to ``cv2.INTER_LINEAR``)
            # ``area`` (identical to ``cv2.INTER_AREA``)
            # ``cubic`` (identical to ``cv2.INTER_CUBIC``)
            image_batch = ia.imresize_many_images(image_batch, self.resize_shape, interpolation='linear')

        label_batch = image_batch
        aug_batch = image_batch
        if self.augment == True:
            label_batch = self.aug_seq.augment_images(image_batch)

            # aug_batch = random_mask.augment_images(label_batch)


        return label_batch,label_batch

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __load_image(self,image_path_batch):
        image_batch = []

        for path in image_path_batch:

            image = cv2.imread(path)
            image = cv2.resize(image,dsize=self.resize_shape)
            image = np.float32(image)
            image /= 255.0
            # image = (image - np.mean(image))/np.std(image)
            image_batch.append(image)

        image_batch = np.array(image_batch)

        return image_batch




