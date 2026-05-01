import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model.keras")

def predict(data):
    arr = np.array(data).reshape(1, -1)
    result = model.predict(arr)[0][0]
    return float(result), int(result > 0.5)