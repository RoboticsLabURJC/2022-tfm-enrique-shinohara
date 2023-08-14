import os
import sys
import time
import h5py
import argparse
import datetime

import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.dataset import get_augmentations, DatasetSequence
from utils.processing import process_dataset
from utils.pilotnet import pilotnet_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.keras.models import load_model

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)


class ScatterPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, annotations_val, images_val):
        self.annotations_val = annotations_val
        self.images_val = images_val
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print('Saving Scatter Plot...')
            
            # Scatter plot of the steering values
            x_true = []
            y_predicted = []

            for x in progressbar(range(0, len(self.annotations_val), 20), "Computing: ", 40):
                x_true.append(self.annotations_val[x][1])

                final_image = self.images_val[x]
                final_image = final_image[np.newaxis]
                prediction = model.predict(final_image)
                # print(prediction)

                y_predicted.append(prediction[0][1])

            fig1, ax1 = plt.subplots()
            ax1.scatter(x_true, y_predicted, c ="green",
                    linewidths = 2,
                    marker =".",
                    s = 50)
        
            plt.xlabel("steer groundtruth")
            plt.ylabel("steer prediction")
            name = 'scatter_plot_10+curves+weird_extreme_epoch' + str(epoch)
            fig1.savefig(name, bbox_inches='tight')
            plt.close(fig1)



class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, annotations_val, images_val, model):
        self.annotations_val = annotations_val
        self.images_val = images_val
        self.model = model
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print('Saving Plot...')
            
            # Scatter plot of the steering values
            x_true = []
            y_predicted = []

            # Scatter plot of the throttle values
            x_true_throttle = []
            y_predicted_throttle = []

            for x in progressbar(range(0, len(self.annotations_val), 50), "Computing: ", 40):
                x_true.append(self.annotations_val[x][1])
                x_true_throttle.append(self.annotations_val[x][0])

                final_image = self.images_val[x] / 255.0
                # print(final_image.shape)
                """velocity_dim = np.full((66, 200), self.annotations_val[x][2])
                final_image = np.dstack((final_image, velocity_dim))"""

                final_image = final_image[np.newaxis]
                prediction = self.model.predict(final_image)

                y_predicted.append(prediction[0][1])
                y_predicted_throttle.append(prediction[0][0])
                # y_predicted.append(prediction[0])
                
            # Steering values plot
            fig1, ax1 = plt.subplots(figsize=(20, 10))
            ax1.plot(x_true, c ="green")
            ax1.plot(y_predicted, c ="red")
        
            plt.xlabel("instance")
            plt.ylabel("steering")
            name = 'plot_graph_steering_epoch' + str(epoch)
            fig1.savefig(name, dpi = 100)
            plt.close(fig1)

            # Throttle values plot
            fig1, ax1 = plt.subplots(figsize=(20, 10))
            ax1.plot(x_true_throttle, c ="green")
            ax1.plot(y_predicted_throttle, c ="red")
        
            plt.xlabel("instance")
            plt.ylabel("throttle")
            name = 'plot_graph_throttle_epoch' + str(epoch)
            fig1.savefig(name, dpi = 100)
            plt.close(fig1)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Data")
    parser.add_argument("--preprocess", action='append', default=None,
                        help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--data_augs", type=int, default=0, help="Data Augmentations: 0=No / 1=Normal / 2=Normal+Weather changes")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of Epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for Policy Net")
    parser.add_argument("--img_shape", type=str, default=(200, 66, 3), help="Image shape")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    path_to_data = args.data_dir[0]
    preprocess = args.preprocess
    data_augs = args.data_augs
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    img_shape = tuple(map(int, args.img_shape.split(',')))

    if 'no_crop' in preprocess:
        type_image = 'no_crop'
    else:
        type_image = 'crop'

    if 'extreme' in preprocess:
        data_type = 'extreme'
    else:
        data_type = 'no_extreme'

    images_train, annotations_train, images_val, annotations_val = process_dataset(path_to_data, type_image,
                                                                                               data_type, img_shape)

    # Train
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(timestr)

    hparams = {
        'batch_size': batch_size,
        'n_epochs': num_epochs,
        'checkpoint_dir': '../logs_test/'
    }

    print(hparams)

    model_name = 'pilotnet_model'
    model = pilotnet_model(img_shape, learning_rate)
    model_filename = timestr + '_pilotnet_model_3_' + str(num_epochs)
    model_file = model_filename + '.h5'


    AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST = get_augmentations(data_augs)

    # Training data
    train_gen = DatasetSequence(images_train, annotations_train, hparams['batch_size'],
                                augmentations=AUGMENTATIONS_TRAIN)

    # Validation data
    valid_gen = DatasetSequence(images_val, annotations_val, hparams['batch_size'],
                                augmentations=AUGMENTATIONS_TEST)


    # Load model
    """model = load_model('20230531-091850_pilotnet_model_3_71_cp.h5')
    pre_score = model.evaluate(valid_gen, verbose=0)
    print('Test loss: ', pre_score[0])
    print('Test mean squared error: ', pre_score[1])
    print('Test mean absolute error: ', pre_score[2])"""


    # Define callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    earlystopping = EarlyStopping(monitor="mae", patience=30, verbose=1, mode='auto')
    checkpoint_path = model_filename + '_cp.h5'
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, monitor='mse', save_best_only=True, verbose=1)
    csv_logger = CSVLogger(model_filename + '.csv', append=True)
    plot_callback = PlotCallback(annotations_val, images_val, model)

    # Print layers
    print(model)
    model.build(img_shape)
    print(model.summary())

    # Training
    model.fit(
        train_gen,
        epochs=hparams['n_epochs'],
        verbose=1,
        validation_data=valid_gen,
        # workers=2, use_multiprocessing=False,
        callbacks=[tensorboard_callback, cp_callback, csv_logger, plot_callback]
        )

    # Save model
    model.save(model_file)

    # Evaluate model
    score = model.evaluate(valid_gen, verbose=0)

    print('Evaluating')
    print('Test loss: ', score[0])
    print('Test mean squared error: ', score[1])
    print('Test mean absolute error: ', score[2])

    model_path = model_file
    # Save model metadata
    with h5py.File(model_path, mode='w') as f:
        hdf5_format.save_model_to_hdf5(model, f)
        f.attrs['experiment_name'] = ''
        f.attrs['experiment_description'] = ''
        f.attrs['batch_size'] = hparams['batch_size']
        f.attrs['nb_epoch'] = hparams['n_epochs']
        f.attrs['model'] = model_name
        f.attrs['img_shape'] = img_shape
        f.attrs['normalized_dataset'] = True
        f.attrs['sequences_dataset'] = True
        f.attrs['gpu_trained'] = True
        f.attrs['data_augmentation'] = True
        f.attrs['extreme_data'] = False
        f.attrs['split_test_train'] = 0.30
        f.attrs['instances_number'] = len(annotations_train)
        f.attrs['loss'] = score[0]
        f.attrs['mse'] = score[1]
        f.attrs['mae'] = score[2]
        f.attrs['csv_path'] = model_filename + '.csv'
