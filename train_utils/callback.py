from keras.callbacks import *


class CustomTensorboard(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 update_freq='epoch'):

        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.batch_size = batch_size
        self.write_grads = write_grads
        self.write_graph = write_graph
        self.write_images = write_images
        self.update_freq = update_freq

        super(CustomTensorboard, self).__init__(
            log_dir=self.log_dir,
            histogram_freq=self.histogram_freq,
            batch_size=self.batch_size,
            write_graph=self.write_graph,
            write_grads=self.write_grads,
            write_images=self.write_images,
            update_freq=self.update_freq)
