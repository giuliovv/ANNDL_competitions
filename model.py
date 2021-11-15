import os
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, path):
        self.path = path
        pass

    def predict(self, X):
        """
        X: numpy.ndarray of shape (1, H, W, C) containing the dataset
        """
        model = tf.keras.load(self.path+"model")
        res = model.predict(X.reshape(X.shape[1], X.shape[2], X.shape[3]))
        return np.argmax(res)