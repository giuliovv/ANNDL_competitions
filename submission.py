import os
import tensorflow as tf

class Model:
    def __init__(self, path):
        self.path = path
        pass

    def predict(self, X):
        """
        X: numpy.ndarray of shape (n, d) containing the dataset
        """
        model = tf.keras.load(self.path+"model")
        model.predict(X)
        pass