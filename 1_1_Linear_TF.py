import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf 
from tensorflow.keras import layers, models
print(f"Imports complete")

model = models.Sequential()
model.add(layers.Dense(256, activation='relu',input_shape=(512,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))

print(model.summary())
