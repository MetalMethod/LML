
"""
Igor Busquets LML
"""

# CONVOLUTION NEURAL NETWOrKS

# Import TensorFlow
import tensorflow
import keras

# Data preprocessing: separte images in categories with folders and files

####### Part 1 - Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

# Step 1 - Add first convolution layer
# generates the feature maps
# params: 
# nb_filters = number of feature detectors and number of feature maps created. 
#32 is very used with first layers

# nb_rows (size of feature fetecture in pixels)
# number of columns (size of feature fetecture in pixels)
# 3 x 3 feature maps
#input_shape = shape of the images expected, in pixels in 3 channels RGB, usually is 256 x 256. 
#channels = 3
#format 64 x 64
# MUST HAVE THE INPUT SHAPE FIRST IN TENSORFLOW
#activation function = ReLU to remove the negative values
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu' ))

# STep 2 - Pooling