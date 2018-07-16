from tensorflow import keras
import tensorflow as tf
from keras.datasets import mnist
from keras import models
from keras import layers
from keras import backend
from keras.utils import to_categorical

# constants
BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 5

# dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

if backend.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

# restrict pixel values
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# categorize labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential([
    layers.Flatten(input_shape = input_shape), 
    layers.Dense(512, activation = 'relu'), 
    layers.Dense(NUM_CLASSES, activation = 'softmax')
])

model.compile(optimizer = 'rmsprop', 
              loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(x = train_images, y = train_labels, 
          epochs = NUM_EPOCHS, 
          batch_size = BATCH_SIZE, 
          verbose = 1, 
          validation_data = (test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 0)
print('Test accuracy: ' + str(test_acc))

# save the model's configuration to avoid re-training each time
model_json = model.to_json()

with open('model_config.json', 'w') as json_file:
    json_file.write(model_json)
    model.save_weights('model.h5')