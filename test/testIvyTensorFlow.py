import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# Let's use the MNIST data set example from TF for image recognition
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Cast the records into float values
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize image pixel values by dividing
# by 255
gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale


print("Feature matrix:", x_train.shape)
print("Target matrix:", x_test.shape)
print("Feature label matrix:", y_train.shape)
print("Target label matrix:", y_test.shape)

npix_x = x_train.shape[1]
npix_y = x_train.shape[2]

# Visualize the training
fig,panels = plt.subplots(10, 10)
k = 0
for i in range(10):
   for j in range(10):
      panels[i][j].imshow(x_train[k].reshape(npix_x, npix_y), aspect='auto')
      k += 1
#fig.set_tight_layout(True)
fig.savefig("test_ivymlpimage_mnist.png")


# Build the MLP training model
model = Sequential([
   # reshape data to flattened rows
   Flatten(input_shape=(npix_x, npix_y)),
   # dense layer 1
   Dense(256, activation='relu'),
   # dense layer 2
   Dense(128, activation='relu'),
   # output layer
   Dense(10, activation='sigmoid')
])
model.summary()

model.compile(
   optimizer='adam',
   loss='sparse_categorical_crossentropy',
   metrics=['accuracy']
)
model.fit(
   x_train, y_train, epochs=10,
   batch_size=2000,
   validation_split=0.2
)

results = model.evaluate(x_train,  y_train, verbose = 0)
print('train loss, train acc:', results)
results = model.evaluate(x_test,  y_test, verbose = 0)
print('test loss, test acc:', results)

# Let's do the same with a convolutional NN
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

print("Feature matrix:", train_images.shape)
print("Target matrix:", test_images.shape)
print("Feature label matrix:", train_labels.shape)
print("Target label matrix:", test_labels.shape)
npix_x = train_images.shape[1]
npix_y = train_images.shape[2]
ncols = train_images.shape[3]
# Visualize the training
fig,panels = plt.subplots(10, 10)
k = 0
for i in range(10):
   for j in range(10):
      panels[i][j].imshow(train_images[k].reshape(npix_x, npix_y, ncols), aspect='auto')
      k += 1
#fig.set_tight_layout(True)
fig.savefig("test_ivymlpimage_cifar10.png")

model = Sequential([
   Conv2D(32, (3, 3), activation='relu', input_shape=(npix_x, npix_y, ncols)),
   MaxPooling2D((2, 2)),
   Conv2D(64, (3, 3), activation='relu'),
   MaxPooling2D((2, 2)),
   Conv2D(64, (3, 3), activation='relu'),
   Flatten(),
   Dense(64, activation='relu'),
   Dense(10)
])
model.summary()

model.compile(
   optimizer='adam',
   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
   metrics=['accuracy']
)

history = model.fit(
   train_images, train_labels, epochs=10,
   validation_data=(test_images, test_labels)
)
results = model.evaluate(train_images, train_labels, verbose=0)
print('train loss, train acc:', results)
results = model.evaluate(test_images, test_labels, verbose=0)
print('test loss, test acc:', results)
