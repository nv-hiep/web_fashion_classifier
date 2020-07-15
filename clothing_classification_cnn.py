# Import
import numpy             as np
import matplotlib.pyplot as plt

from keras.datasets      import fashion_mnist

# MNIST fashion dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Just to check
# Training set
print("Training set")
print("Number of samples: " + str(len(x_train)))
print("Number of labels: " + str(len(y_train)))
print("Dimensions of a single image:" + str(x_train[0].shape))

print("-------------------------------------------------------------")

# Test set
print("Test set")
print("Number of samples: " + str(len(x_test)))
print("Number of labels: " + str(len(y_test)))
print("Dimensions of single image:" + str(x_test[0].shape))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""### Some sample images"""

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()

"""### Let's create our model"""

#Import necessary keras specific libraries

import keras
from keras.utils  import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras        import backend as K

# Setting Training Parameters like batch_size, epochs
batch_size = 128
epochs     = 100

# Storing the number of rows and columns
img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

''' Getting the data in the right 'shape' as required by Keras i.e. adding a 4th 
dimension to our data thereby changing the original image shape of (60000,28,28) 
to (60000,28,28,1)'''
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test  = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Storing the shape of a single image, 28 x 28 x 1
input_shape = (img_rows, img_cols, 1)

# Changing image type to float32 data type
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

# Normalizing the data by changing the image pixel range from (0 to 255) to (0 to 1)
x_train = x_train / 255.
x_test  = x_test / 255.

# Performing one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)

# Calculate the number of classes and number of pixels 
num_classes = y_test.shape[1]
num_pixels  = x_train.shape[1] * x_train.shape[2]

# Create CNN model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])

print(model.summary())

"""## Train the model"""

model_fitting = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Mount the google drive
from google.colab import drive
drive.mount('/content/drive')

# Change the directory to current working directory
import os
os.chdir("/content/drive/My Drive/data")

# Save the model with the name clothing_classification_model
model.save('fashion_classification_model.h5')

# Import few more necessary libraries.
from keras.models              import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Function to load and prepare the image in right shape
def load_image(filename):
	# Load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# Convert the image to array
	img = img_to_array(img)
	# Reshape the image into a sample of 1 channel
	img = img.reshape(1, 28, 28, 1)
	# Prepare it as pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img


filename = 'ankle_boot.jpg'
filename = 'shirt.jpeg'
filename = 'sandal.jpg'
img = load_img(filename, grayscale=True, target_size=(28, 28))
plt.imshow(img, cmap=plt.cm.binary)
plt.show()

# Load an image and predict the apparel class
img = load_image(filename)
# Load the saved model
# model = load_model('fashion_classification_model.h5')
model = load_model('model.h5')
# Predict the apparel class
class_prediction = model.predict_classes(img)
print(class_prediction)
pred_id = class_prediction[0]

# Map apparel category with the numerical class
print('All classes: ')
print(class_names)
print('predicted class: ')
print(class_names[pred_id])