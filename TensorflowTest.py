import sys
import cv2
import numpy as np
import pandas as pd
from tflite_runtime.interpreter import Interpreter

print("Python version = " + sys.version)
print("cv2 version = " + cv2.__version__)
print("numpy version = " + np.__version__)
print("pandas version = " + pd.__version__)

import tensorflow as tf
from tensorflow import keras

print("tensorflow version = " + tf.__version__)
print("keras version = " + keras.__version__)