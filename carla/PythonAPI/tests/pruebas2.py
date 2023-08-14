import tensorflow as tf
import numpy as np

"""TAG_NAME = "classification_loss"
# event_file = '/home/yujiro/git/2022-tfm-enrique-shinohara/carla/PythonAPI/tests/training/logs/fit/20230516-233043/train/events.out.tfevents.1684272643.yujiro-L-5-15ACH6Hegion.695100.0.v2'
event_file = '/home/yujiro/git/2022-tfm-enrique-shinohara/carla/PythonAPI/tests/training/logs/fit/20230516-233043/validation/events.out.tfevents.1684272741.yujiro-L-5-15ACH6Hegion.695100.1.v2'

epoch_loss_values = []

# Iterate over events in the event file
for event in tf.compat.v1.train.summary_iterator(event_file):
    for value in event.summary.value:
        if value.tag == 'epoch_loss':
            epoch_loss_values.append(tf.make_ndarray(value.tensor))

epoch_loss_array = np.array(epoch_loss_values)
np.save('validation_epoch_loss.npy', epoch_loss_array, allow_pickle=True)"""


training = np.load('training_epoch_loss.npy', allow_pickle=True)
validation = np.load('validation_epoch_loss.npy', allow_pickle=True)

print(training)
print(len(training))
print(validation)
print(len(validation))