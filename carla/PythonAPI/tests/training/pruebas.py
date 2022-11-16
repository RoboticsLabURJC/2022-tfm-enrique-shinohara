import os
import sys
import cv2 as cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

images_val = np.load('array_imgs.npy', allow_pickle=True)
annotations_val = np.load('array_annotations.npy', allow_pickle=True)
model = load_model('20221105-111107_pilotnet_model_3_51_cp.h5')

# Scatter plot of the steering values
x_true = []
y_predicted = []

# Scatter plot of the throttle values
x_true_throttle = []
y_predicted_throttle = []

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


for x in progressbar(range(0, len(annotations_val), 100), "Computing: ", 40):
    x_true.append(annotations_val[x][1])
    x_true_throttle.append(annotations_val[x][0])
    
    final_image = images_val[x]
    final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    final_image = final_image / 255

    final_image = final_image[np.newaxis]
    
    prediction = model.predict(final_image)

    y_predicted.append(prediction[0][1])
    y_predicted_throttle.append(prediction[0][0])
    # y_predicted.append(prediction[0])
    
# Steering values plot
fig1, ax1 = plt.subplots(figsize=(20, 10))
ax1.plot(x_true, c ="green")
ax1.plot(y_predicted, c ="red")

plt.xlabel("instance")
plt.ylabel("steering")
name = 'plot_graph_steering'
fig1.savefig(name, dpi = 100)
plt.close(fig1)

# Throttle values plot
fig1, ax1 = plt.subplots(figsize=(20, 10))
ax1.plot(x_true_throttle, c ="green")
ax1.plot(y_predicted_throttle, c ="red")

plt.xlabel("instance")
plt.ylabel("throttle")
name = 'plot_graph_throttle'
fig1.savefig(name, dpi = 100)
plt.close(fig1)