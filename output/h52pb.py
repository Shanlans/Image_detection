
import os

import tensorflow as tf

from keras import backend as K
from keras.models import load_model

def convert_h5_to_pb(dir, filename):
    model = load_model(os.path.join(dir, filename), compile=False)
    name = 'saved_model.pb'
    # Function 1
    output_names = [out.op.name for out in model.outputs]
    # Freezes the state of a session into a pruned computation graph.
    graph = K.get_session().graph
    with graph.as_default():
        input_graph_def = graph.as_graph_def()
    for node in input_graph_def.node:
        node.device = ""
    frozen_graph = tf.graph_util.convert_variables_to_constants(K.get_session(), input_graph_def,
                                                             output_names)
    with tf.gfile.GFile(os.path.join(dir, name), "wb") as f:
        f.write(frozen_graph.SerializeToString())
    # Function 2
    # export_path = os.path.join(dir, 'saved_model2')
    # with K.get_session() as sess:
    #     tf.saved_model.simple_save(
    #         sess,
    #         export_path,
    #         inputs={'input_image': model.input},
    #         outputs={t.name: t for t in model.outputs}
    #     )

convert_h5_to_pb('../logs/models','weights-improvement-16-0.99.hdf5')
