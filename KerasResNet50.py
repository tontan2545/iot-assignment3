# TensorFlow and tf.keras
import tensorflow as tf
from keras_applications import resnet50

# Load the ResNet50 model
resnet50_model = resnet50.ResNet50(weights='imagenet')

# Save model and Export
savepath = "Resnet50.h5"
resnet50_model.save(savepath)