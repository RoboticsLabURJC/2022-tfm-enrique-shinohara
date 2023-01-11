from tensorflow.keras.applications import VGG16
from tensorflow import keras

scale=1
img_size  = (224,224,3)

model = VGG16(input_shape = img_size,
                include_top = True,
                weights     = 'imagenet')

preprocess_input   = keras.applications.vgg16.preprocess_input
decode_predictions = keras.applications.vgg16.decode_predictions

last_conv_layer_name   = 'block5_conv3'
classifier_layer_names =  ['block5_pool', 'flatten', 'fc1', 'fc2',"predictions",]

print(model.summary())