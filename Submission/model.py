import os
import tensorflow as tf

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel/model1_20'))
        return

    def predict(self, X):
        X = tf.keras.applications.vgg16.preprocess_input(X.numpy())
        prediction = self.model.predict(X)
        return tf.argmax(prediction, axis=-1)