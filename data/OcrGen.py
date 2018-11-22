import os
import math
import numpy as np
import imgaug as ia
import pandas as pd
import cv2

from keras.utils import np_utils, Sequence
from keras.datasets import cifar10, cifar100



class OCRDataGen(Sequence):

    def __init__(self, data_path,characterset,fixed_length,batch_size, resize_shape, train_phase=True, shuffle=False,
                 use_cache=True, augment=False, aug_seq=None):
        """

        :param data_path: A string, point to data classification data path
        :param characterset: A String, E.g '0123456789+-*()' (15 characters)
        :param fixed_length: A Integer. This OCR data generator only support the fixed length ocr
        :param batch_size: An Integer, Batch size for training
        :param resize_shape: A Tuple. (Width,Height), image should be resize to the shape which matches model input shape
        :param train_phase: Boolean, Default is 'True' which will return the training data (images and labels),
                            otherwise return the validation data (images and labels)
        :param shuffle: Boolean. Default is 'False' which would not shuffle data set, otherwise shuffle it
        :param use_cache: Boolean, Default is 'True' which cache the every image when the first epochs, second epochs,
                            no need to reload image cause it will waste the time do I/O operation
        :param augment: Boolean. Default is 'False' which would not do augmentation for this batch of data
        :param aug_seq: ImgAug Sequential Object. Default is None. A simple sequence can follow this guide https://imgaug.readthedocs.io/en/latest/source/examples_basics.html#a-simple-and-common-augmentation-sequence
        """

        self.data_path = os.path.join(data_path,'label.csv')
        self.images_path = os.path.join(data_path,'image')
        self.characterset = characterset
        self.fixed_length = fixed_length
        self.batch_size = batch_size
        self.resize_shape = resize_shape
        self.train_phase = train_phase
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        self.aug_seq = aug_seq

        # Get character encoder and decoder dict
        self.encode_maps = {}
        self.decode_maps = {}
        self.class_name = []
        self.__character_coder()

        self.__load_data()

        self.images = self.orig_images
        self.labels = self.orig_labels

        if use_cache == True:
            # Cache Image
            # cache: Initial a empty numpy array with image shape and total image number
            # is_cached: Initial a flag array with image number to indicate whether image was cached

            self.cache = np.zeros_like(self.images, dtype=np.float32)
            self.is_cached = np.zeros(len(self.images))
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

        image_batch = self.__cache_data(indexes, image_batch)

        # One-hot label, E.g. if self.class_num = 5
        # Before: [[1,2,...],[3,4,...]...]
        # After: [[[0,1,0,0,0],[0,0,1,0,0]...] , [[0,0,0,1,0],[0,0,0,0,1]...]...]
        label_batch = np_utils.to_categorical(self.labels[indexes], self.class_num)

        # For the multi label fixed-length OCR problem, the label shape must be [BATCH,FIXED_LENGTH*CLASS]
        label_batch = np.reshape(label_batch,(-1,self.fixed_length*self.class_num))

        if self.resize_shape:
            # Resize images interpolation choice
            # ``nearest`` (identical to ``cv2.INTER_NEAREST``)
            # ``linear`` (identical to ``cv2.INTER_LINEAR``)
            # ``area`` (identical to ``cv2.INTER_AREA``)
            # ``cubic`` (identical to ``cv2.INTER_CUBIC``)
            image_batch = ia.imresize_many_images(image_batch, self.resize_shape, interpolation='linear')
            image_batch = image_batch

        if self.augment == True:
            image_batch = self.aug_seq.augment_images(image_batch)

        return image_batch, label_batch

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __character_coder(self):
        for i, char in enumerate(self.characterset, 1):
            self.class_name.append(char)
            self.encode_maps[char] = i-1
            self.decode_maps[i-1] = char



    def __load_data(self):
        df = pd.read_csv(self.data_path)
        images = []
        labels = []

        for name in df[df.columns[0]].values:
            image_path = os.path.join(self.images_path,name)
            images.append(cv2.imread(image_path))

        # labels is an array                       E.g [['A B...'],['C D...']...]
        # 1st. Convert this array to list          E.g [['A','B',...],['C','D',...]...]
        # 2nd. Convert this array with encode_maps E.g [[1,2...],[3,4...]...]
        for label in df[df.columns[1]].values:
            labels.append([self.encode_maps[x] if x in self.encode_maps else x for x in label.split(' ')])

        self.class_num = len(self.encode_maps.keys())

        self.orig_images = np.array(images)

        self.orig_labels = np.array(labels)








    def __cache_data(self, indexes, images):
        cache_batch_image = np.zeros_like(images)
        if self.use_cache == True:

            # From 'cache' load images using 'indexes', if not cached, will load 'zeros'
            cache_batch_image = self.cache[indexes]

            # Find which images was not cached in this batch.
            batch_not_cached = np.where(self.is_cached[indexes] == 0)

            for i, image in enumerate(images[batch_not_cached]):
                # Flag set to 1 to indicate this image has already been cached this time
                self.is_cached[indexes[i]] = 1

                self.cache[indexes[i]] = image

                cache_batch_image[i] = image
        else:
            for i, image in enumerate(images):
                cache_batch_image[i] = image

        return cache_batch_image
