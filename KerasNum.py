# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#load data set from tf.keras.datasets.mnist (Train number and Test number)
mnist = tf.keras.datasets.mnist # 28x28 image of hand-written digits 0-9
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# Build Model
model = keras.Sequential([
keras.layers.Flatten(input_shape=(28, 28)), # takes our 28x28 and makes it 1x784
keras.layers.Dense(128, activation='relu'), # a simple fully-connected layer, 128 units, relu activation
keras.layers.Dense(128, activation='relu'), # a simple fully-connected layer, 128 units, relu activation
keras.layers.Dense(10, activation='softmax') # our output layer. 10 units for 10 classes. Softmax for probability distribution
])
# Compile model
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train, epochs=5)
# Evaluate accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
# ***Save model***
savepath = "num_reader.h5"
model.save(savepath)