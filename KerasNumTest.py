# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import cv2
# load an image
Testing_img_path = "./Testing_image/number3.jpg"
input_image = cv2.imread(Testing_img_path,cv2.IMREAD_GRAYSCALE)
# Preprae data
input_image = cv2.resize(input_image,(28, 28))
input_image = cv2.bitwise_not(input_image) # invert Black to White like data input when trained (Line is White)
input_image = np.expand_dims(input_image, axis=0) # Change the shape of image array like input image when trained
# ***load model***
loadpath = "num_reader.h5"
model = tf.keras.models.load_model(loadpath)
# Make predictions
predictions = model.predict(input_image)
print(np.argmax(predictions))
# Visualize, check the answer
Original_input_image_testing = cv2.imread(Testing_img_path)
cv2.imshow("This is what your computer has seen.",Original_input_image_testing)
cv2.waitKey(0)
cv2.destroyAllWindows()