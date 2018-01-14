
"""
Igor Busquets LML
"""

# CONVOLUTION NEURAL NETWOrKS

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

# Step 2 - Pooling
#using subtable of 2x2 size generate a pooled feature map

classifier.add(MaxPooling2D(pool_size = (2, 2)))

###SECOND CONVOLUTIONAL LAYER
classifier.add(Convolution2D(32, 3, 3, activation = 'relu' ))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection hidden layer by ANN using falt input
#Hidden layer
#output_dim = number of hidden nodes betweeen the input number and output nodes number
classifier.add(Dense(output_dim = 128, activation = 'relu'))
#Output layer - sigmoid because its binary output, if it where ore than 1 class, it would be Softmax
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compile the CNN
#Using Stochastic Gradient Descent
#Loss function
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Image preprocessing to fit the classifier to different sizes images
#Keras image augmentation - avoids ovefitting: great results in training set but low results in testset
#from keras documentation: 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifier.fit_generator(training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        nb_val_samples=2000)