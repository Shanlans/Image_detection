

from models import *
from data_utils import *
from config import *


flags = tf.app.flags

FLAGS = flags.FLAGS

epochs = 100


def main(argv):

    data = FasterRcnnDataGen(data_dir=FLAGS.data_dir,batch_size=2,front_end='VGG16')



    # model = BuildModel(input_shape=(1000,600,3),front_end='VGG16',model_name=None,job_type=None,initial_weights=None)




if __name__ == "__main__":
    # set_seed(666)
    tf.app.run()
