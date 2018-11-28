import math
import numpy as np
import imgaug as ia

from keras.utils import np_utils, Sequence
from keras.datasets import cifar10, cifar100

CIFAR10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

CIFAR100 = ["beaver", "dolphin", "otter", "seal", "whale",
              "aquarium fish", "flatfish", "ray", "shark", "trout",
              "orchids", "poppies", "roses", "sunflowers", "tulips",
              "bottles", "bowls", "cans", "cups", "plates",
              "apples", "mushrooms", "oranges", "pears", "sweet peppers",
              "clock", "computer keyboard", "lamp", "telephone", "television",
              "bed", "chair", "couch", "table", "wardrobe",
              "bee", "beetle", "butterfly", "caterpillar", "cockroach",
              "bear", "leopard", "lion", "tiger", "wolf",
              "bridge", "castle", "house", "road", "skyscraper",
              "cloud", "forest", "mountain", "plain", "sea",
              "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
              "fox", "porcupine", "possum", "raccoon", "skunk",
              "crab", "lobster", "snail", "spider", "worm",
              "baby", "boy", "girl", "man", "woman",
              "crocodile", "dinosaur", "lizard", "snake", "turtle",
              "hamster", "mouse", "rabbit", "shrew", "squirrel",
              "maple", "oak", "palm", "pine", "willow",
              "bicycle", "bus", "motorcycle", "pickup truck", "train",
              "lawn-mower", "rocket", "streetcar", "tank", "tractor"]


class CIFARDataGen(Sequence):

    def __init__(self, batch_size, resize_shape, cifa_10=True, train_phase=True, shuffle=False,
                 use_cache=True, augment=False, aug_seq=None):
        """
        :param batch_size: An Integer, Batch size for training
        :param resize_shape: A Tuple. (Width,Height), image should be resize to the shape which matches model input shape
        :param cifa_10: Boolean. Default is 'True' which means loading the cifa-10 dataset,
                        otherwise using the cifa-100 dataset
        :param train_phase: Boolean, Default is 'True' which will return the training data (images and labels),
                            otherwise return the validation data (images and labels)
        :param shuffle: Boolean. Default is 'False' which would not shuffle data set, otherwise shuffle it
        :param use_cache: Boolean, Default is 'True' which cache the every image when the first epochs, second epochs,
                            no need to reload image cause it will waste the time do I/O operation
        :param augment: Boolean. Default is 'False' which would not do augmentation for this batch of data
        :param aug_seq: ImgAug Sequential Object. Default is None. A simple sequence can follow this guide https://imgaug.readthedocs.io/en/latest/source/examples_basics.html#a-simple-and-common-augmentation-sequence
        """

        self.batch_size = batch_size
        self.resize_shape = resize_shape
        self.cifa_10 = cifa_10
        self.train_phase = train_phase
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        self.aug_seq = aug_seq

        self.images_shape = (resize_shape[0],resize_shape[1],3)

        if self.cifa_10:
            self.orig_images, self.orig_labels = self.__load_cifa10()
            self.class_num = 10
            self.class_name = CIFAR10
        else:
            self.orig_images, self.orig_labels = self.__load_cifa100()
            self.class_num = 100
            self.class_name = CIFAR100

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

        # One-hot label
        label_batch = np_utils.to_categorical(self.labels[indexes], self.class_num)

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

    def __load_cifa10(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        if self.train_phase == True:
            return x_train, y_train
        else:
            return x_test, y_test

    def __load_cifa100(self):
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        if self.train_phase == True:
            return x_train, y_train
        else:
            return x_test, y_test

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
