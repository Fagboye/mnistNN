
import tensorflow as tf
from tensorflow import keras


# loading and preprocessing of data
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# normalization of training images and test images to range of 0 to 1 for sable and efficient learning
train_images, test_images = train_images / 255.0, test_images / 255.0

## definition of the neural network architecture
model = keras.Sequential ([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(123, activation = 'relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation = 'softmax')
])

## compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

## Training the model
history = model.fit(train_images, train_labels, epochs=6, validation_data=(test_images, test_labels))

## evaluation of model
test_loss, test_accuracy = model.evaluate(test_images, test_labels,verbose=2)
print(f"test accuracy: {test_accuracy*100:.2f}%")

# save model 
model.save('minst.h5')

