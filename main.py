

from models import *
from data_utils import *
from train_utils import *
from keras.optimizers import *

from config import *

flags = tf.app.flags

FLAGS = flags.FLAGS


def main(argv):

    train_gen = FasterRcnnDataGen(data_dir=FLAGS.data_dir,front_end='VGG16',train_phase=True)
    valide_gen = FasterRcnnDataGen(data_dir=FLAGS.data_dir,front_end='VGG16',train_phase=False)

    buildModel = BuildModel(front_end=FLAGS.front_end,model_name=FLAGS.model_name,job_type=FLAGS.job_type,initial_weights=None)

    model_rpn = buildModel.build_model()

    optimizer = Adam(lr=FLAGS.base_learning_rate)

    model_rpn.compile(optimizer=optimizer,
                      loss=[rpn_loss_cls(buildModel.num_anchors), rpn_loss_regr(buildModel.num_anchors)])

    tensorboard = FasterRcnnTensorboard(log_dir='./logs',histogram_freq=1,validation_data=valide_gen,batch_size=FLAGS.batch_size,write_images=True,update_freq='batch')

    hist = model_rpn.fit_generator(train_gen,steps_per_epoch=len(train_gen),epochs=FLAGS.epochs,validation_data=valide_gen,validation_steps=len(valide_gen),callbacks=[tensorboard])











if __name__ == "__main__":
    # set_seed(666)
    tf.app.run()
