
import tensorflow as tf

from keras import backend as K
from keras.callbacks import *
from keras.utils import Sequence

from monitor import cm_analysis


class CustomTensorboard(TensorBoard):
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



        super(CustomTensorboard, self).__init__(
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

        super(CustomTensorboard,self).set_model(model)


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

                val = (batch_image, batch_label, batch_sample_weighted)
                if self.model.uses_learning_phase:
                    val += [True]

                assert len(val) == len(tensors)
                feed_dict = dict(zip(tensors, val))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                idx = self.epoch * iter_time + batch
                self.writer.add_summary(summary_str, idx)


        super(CustomTensorboard,self).on_batch_end(batch,logs)


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


                    val = (batch_image,batch_label,batch_sample_weighted)
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
        super(CustomTensorboard,self)._write_logs(logs,index)





class FullEvaluate(Callback):
    def __init__(self,validation_data,batch_size,log_dir='./logs'):

        super(FullEvaluate, self).__init__()
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.validation_data = validation_data
        self.class_num = validation_data.class_num
        self.class_name = validation_data.class_name
        if not isinstance(self.validation_data, Sequence):
            raise ValueError("Validation_data must be provided, and must be a generator (like Sequence obj).")
        # use iter_sequence to avoid the queue is unvaliable
        self.output_generator = iter_sequence_infinite(self.validation_data)

    def set_model(self, model):
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()
        super(FullEvaluate,self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):

        val_data = self.validation_data
        tensors = (self.model.inputs +
                   self.model.targets +
                   self.model.sample_weights)

        if self.model.uses_learning_phase:
            tensors += [K.learning_phase()]

        iter_time = len(val_data)  # iter times
        i = 0

        full_pred = np.empty((0,self.class_num))
        full_true = np.empty((0,self.class_num))
        while i < iter_time:
            if self.model.sample_weight_mode is None:
                batch_sample_weighted = np.ones((self.batch_size,))

            batch_image, batch_label = next(self.output_generator)

            val = (batch_image, batch_label, batch_sample_weighted)
            if self.model.uses_learning_phase:
                val += [False]

            assert len(val) == len(tensors)
            feed_dict = dict(zip(tensors, val))
            pred = self.sess.run([self.model.output], feed_dict=feed_dict)
            full_pred = np.concatenate((full_pred,pred[0]),axis=0)
            full_true = np.concatenate((full_true,batch_label),axis=0)
            i+=1

        y_pred = np.argmax(full_pred,axis=-1)
        y_true = np.argmax(full_true,axis=-1)

        filename = os.path.join(self.log_dir,'{:2d}-{val_loss:.2f}-{val_acc:.2f}.png'.format(epoch,**logs))
        cm_analysis(y_true=y_true,y_pred=y_pred,filename=filename,labels=self.class_name)



class FullEvaluateForMultiLabels(Callback):
    def __init__(self,validation_data,batch_size,log_dir='./logs',threshold_update_fn=None):

        super(FullEvaluateForMultiLabels, self).__init__()
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.validation_data = validation_data
        self.class_num = validation_data.class_num
        self.fixed_length = validation_data.fixed_length
        self.class_name = validation_data.class_name
        self.threshold_update_fn = threshold_update_fn
        if not isinstance(self.validation_data, Sequence):
            raise ValueError("Validation_data must be provided, and must be a generator (like Sequence obj).")
        # use iter_sequence to avoid the queue is unvaliable
        self.output_generator = iter_sequence_infinite(self.validation_data)

    def set_model(self, model):
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()
        super(FullEvaluateForMultiLabels,self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):

        val_data = self.validation_data
        tensors = (self.model.inputs +
                   self.model.targets +
                   self.model.sample_weights)

        if self.model.uses_learning_phase:
            tensors += [K.learning_phase()]

        iter_time = len(val_data)  # iter times
        i = 0

        full_pred = np.empty((0,self.fixed_length,self.class_num))
        full_true = np.empty((0,self.fixed_length,self.class_num))
        while i < iter_time:
            if self.model.sample_weight_mode is None:
                batch_sample_weighted = np.ones((self.batch_size,))

            batch_image, batch_label = next(self.output_generator)

            val = (batch_image, batch_label, batch_sample_weighted)
            if self.model.uses_learning_phase:
                val += [False]

            assert len(val) == len(tensors)
            feed_dict = dict(zip(tensors, val))
            pred = self.sess.run([self.model.output], feed_dict=feed_dict)

            batch_label = batch_label.reshape((-1,self.fixed_length,self.class_num))
            pred = pred[0].reshape((-1,self.fixed_length,self.class_num))
            full_pred = np.concatenate((full_pred,pred),axis=0)
            full_true = np.concatenate((full_true,batch_label),axis=0)
            i+=1

        # Update f1 score threshold
        if self.threshold_update_fn:
            self.threshold_update_fn(full_true,full_pred,self.class_num)

        # Need to do confusion matrix update
        y_pred = np.argmax(full_pred,axis=-1).reshape(-1)
        y_true = np.argmax(full_true,axis=-1).reshape(-1)
        filename = os.path.join(self.log_dir,'{:2d}-{val_loss:.2f}-{val_acc:.2f}.png'.format(epoch,**logs))
        cm_analysis(y_true=y_true,y_pred=y_pred,filename=filename,labels=self.class_name)

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


