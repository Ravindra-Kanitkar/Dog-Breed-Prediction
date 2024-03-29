from flask import Flask,render_template
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

BATCH_SIZE = 32
IMG_SIZE = 224

def load_model(model_path):
  """
  Loads a saved model from a specified path
  """
  print(f"Loading saved model from : {model_path}")
  model = tf.keras.models.load_model(model_path,custom_objects={"KerasLayer":hub.KerasLayer})
  return model

def process_image(image_path,img_size=IMG_SIZE):
  """
  Takes an image file path and turns the image into a Tensor
  """
  #Read an image file
  image = tf.io.read_file(image_path)
  #Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green,Blue)
  image = tf.image.decode_jpeg(image,channels=3)
  # Convert the colour channel values to from 0-255 to 0-1 values
  image = tf.image.convert_image_dtype(image,tf.float32)
  # Resize the image to our desired value (224,224)
  image = tf.image.resize(image,size=[IMG_SIZE,IMG_SIZE])

  return image

# Create a function to turn data into batches
def create_data_batches(X,y=None,batch_size=BATCH_SIZE,valid_data=False,test_data=False):
  """
  Creates batches of data out of image (X) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle if it's validation data.
  Also accepts test data as input (no labels).
  """
  # If the data is a test dataset, we probably don't have labels
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices(tf.constant(X)) # only filepaths(no labels)
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch
  # If the data is a valid dataset,we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch
  else:
    print("Creating training data batches...")
    # Turn filepaths and labels into Tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
    # Shuffling pathnames and lables before mapping image processor function is faster than shuffle
    data = data.shuffle(buffer_size=len(X))
    data = data.map(get_image_label)
    data_batch = data.batch(BATCH_SIZE)

    return data_batch
def get_pred_label(prediction_probabilities):
  return unique_breeds[np.argmax(prediction_probabilities)]



model_path = "models\dog_breed_prediction_model.h5"
loaded_full_model = load_model(model_path)
custom_path = "images/"
labels_csv = pd.read_csv("labels.csv")
labels = labels_csv["breed"].to_numpy()
unique_breeds = np.unique(labels)


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")
