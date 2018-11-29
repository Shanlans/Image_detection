from models import *
from data_utils import *
from train_utils import *
from keras.optimizers import *
from keras.callbacks import *
from monitor import *
from config import *

flags = tf.app.flags

FLAGS = flags.FLAGS


def main(argv):
    train_gen = AutoEncoderDataGen(data_dir=FLAGS.data_dir + '/train', batch_size=FLAGS.batch_size,
                                   resize_shape=(512, 512), train_phase=True,augment=True)

    val_gen = AutoEncoderDataGen(data_dir=FLAGS.data_dir + '/test', batch_size=1, resize_shape=(512, 512),
                                 train_phase=False, shuffle=False, augment=True)

    buildModel = BuildModel(front_end=FLAGS.front_end, model_name=FLAGS.model_name, job_type=FLAGS.job_type,
                            initial_weights=None)

    model = buildModel.build_model()

    optimizer = Adam(lr=FLAGS.base_learning_rate)

    fscore = F1()
    metrics = ['acc', fscore.f1]
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=metrics)

    tb = CustomTensorboard(log_dir='./logs/tensorboard',
                           histogram_freq=1,
                           validation_data=val_gen,
                           batch_size=FLAGS.batch_size,
                           write_image_output = True,
                           write_images=False,
                           update_freq='batch')
    checkpoint_save = ModelCheckpoint(filepath='./logs/models',
                                      monitor='acc',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='max',
                                      period=1)

    hist = model.fit_generator(train_gen, steps_per_epoch=len(train_gen), epochs=FLAGS.epochs,
                               validation_data=val_gen, validation_steps=len(val_gen),
                               callbacks=[tb, checkpoint_save])

    predictions = model.predict_generator(val_gen, steps=len(val_gen), verbose=1)

    num_images = 5
    random_test_images = np.random.randint(len(predictions), size=num_images)

    from matplotlib.pyplot import plot as plt
    for i, image_idx in enumerate(random_test_images):
        # plot original image
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(val_gen[image_idx].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot reconstructed image
        ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(predictions[image_idx].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    # set_seed(666)
    tf.app.run()
