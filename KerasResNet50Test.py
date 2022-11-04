# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Helper libraries
import numpy as np
import cv2

import matplotlib.pyplot as plt

# load an image
Testing_img_path = "./Testing_image/fox.jpg"
input_image = cv2.imread(Testing_img_path)

# Preprae data
input_image = cv2.resize(input_image,(224, 224))
input_image = np.expand_dims(input_image, axis=0) # Change the shape of image array like input image when trained

# Load the ResNet50 model
loadpath = "Resnet50.h5"
resnet_model = tf.keras.models.load_model(loadpath)
# get the predicted probabilities for each class
predictions = resnet_model.predict(input_image)
print("Predicted Top3 is:", decode_predictions(predictions, top=3)[0])
Answer = decode_predictions(predictions, top=1)[0][0][1]
print("Predicted = ", Answer)
# Visualize, check the answer
img = cv2.imread(Testing_img_path)
cv2.imshow('Answer',img)
#plt.imshow(input_image)
#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()