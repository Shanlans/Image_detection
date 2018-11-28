
import keras.backend as K
import tensorflow as tf

from keras.utils import Sequence
from keras.callbacks import *




class FasterRcnnTensorboard(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 validation_data=None,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 max_image_display=3,
                 update_freq='epoch'):
        """

        :param log_dir:
        :param histogram_freq:
        :param validation_data:
        :param batch_size:
        :param write_graph:
        :param write_grads:
        :param write_images:
        :param max_image_display:
        :param update_freq:
        """
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq

        self.batch_size = batch_size
        self.write_grads = write_grads
        self.write_graph = write_graph
        self.write_images = write_images
        self.max_image_display = max_image_display
        self.update_freq = update_freq
        self.epoch = 0



        super(FasterRcnnTensorboard, self).__init__(
            log_dir=self.log_dir,
            histogram_freq=self.histogram_freq,
            batch_size=self.batch_size,
            write_graph=self.write_graph,
            write_grads=self.write_grads,
            write_images=self.write_images,
            update_freq=self.update_freq)

        self.validation_data = validation_data

        if not isinstance(self.validation_data,Sequence):
            raise ValueError("Validation_data must be provided, and must be a generator (like Sequence obj).")
        # use iter_sequence to avoid the queue is unvaliable
        self.output_generator = iter_sequence_infinite(self.validation_data)


    def set_model(self, model):
        '''

        :param model:
        '''
        self.model = model

        # In this program, we should be always Channel last format for input !!!
        w_img = self.model.inputs[0]

        # w_img tensor must be 4-D with shape [batch_size,height,width,channels]
        assert len(w_img.get_shape().as_list()) == 4,'Should be the 4-D images tensor [batch_size,height,width,channels]'

        tf.summary.image('Input_Image',w_img,max_outputs=self.max_image_display)

        super(FasterRcnnTensorboard,self).set_model(model)


    def on_batch_end(self, batch, logs=None):
        """

        :param batch:
        :param logs:
        """
        if self.update_freq!='epoch':
            if batch % self.histogram_freq == 0:
                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                iter_time = len(val_data)  # iter times
                if self.model.sample_weight_mode is None:
                    batch_sample_weighted = np.ones((self.batch_size,))

                batch_image, batch_label = next(self.output_generator)

                val = (batch_image, batch_label[0],batch_label[1], batch_sample_weighted,batch_sample_weighted)
                if self.model.uses_learning_phase:
                    val += [True]

                assert len(val) == len(tensors)
                feed_dict = dict(zip(tensors, val))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                idx = self.epoch * iter_time + batch
                self.writer.add_summary(summary_str, idx)


        super(FasterRcnnTensorboard,self).on_batch_end(batch,logs)


    def on_epoch_end(self, epoch, logs=None):
        """

        :param epoch:
        :param logs:
        """
        logs = logs or {}
        self.epoch = epoch

        if not isinstance(self.validation_data,Sequence) and self.histogram_freq:
            raise ValueError("If printing histograms, validation_data must be provided, and must be a generator (like Sequence obj).")

        if self.update_freq == 'epoch':
            if self.validation_data is not None and self.histogram_freq:
                if epoch % self.histogram_freq == 0:

                    val_data = self.validation_data
                    tensors = (self.model.inputs +
                               self.model.targets +
                               self.model.sample_weights)

                    if self.model.uses_learning_phase:
                        tensors += [K.learning_phase()]

                    if self.model.sample_weight_mode is None:
                        batch_sample_weighted = np.ones((self.batch_size,))

                    batch_image,batch_label = next(self.output_generator)

                    val = (batch_image, batch_label[0], batch_label[1], batch_sample_weighted, batch_sample_weighted)
                    if self.model.uses_learning_phase:
                        val += [True]

                    assert len(val) == len(tensors)
                    feed_dict = dict(zip(tensors, val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)

        if self.update_freq == 'epoch':
            index = epoch
        else:
            index = self.samples_seen
        self._write_logs(logs, index)

    def _write_logs(self, logs, index):

        # Monitor learning rate
        logs['lr'] = K.eval(self.model.optimizer.lr)
        super(FasterRcnnTensorboard,self)._write_logs(logs,index)


def iter_sequence_infinite(seq):
    """Iterate indefinitely over a Sequence.

    # Arguments
        seq: Sequence object

    # Returns
        Generator yielding batches.
    """
    while True:
        for item in seq:
            yield item