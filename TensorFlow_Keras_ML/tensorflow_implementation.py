# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# tensorflow imports
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.utils import to_categorical

# define folder path
path = "../fonts/"

# list each file in folder
files = os.listdir(path)

# concatenate DFs together
df_list = [pd.read_csv(path+file) for file in files]
df = pd.concat(df_list)

# replace labels with integer values
labels_df = df.pop('font')
labels_df_factorized, label_names = pd.factorize(labels_df)
labels = np.array(labels_df_factorized)

# remove text attribute
df = df.pop('fontVariant')


# ------- IMPLEMENTATION 1 - FEED FORWARD NETWORK (all data) ---------------------
# convert data to NP array
data = np.array(df)
data = data.astype(np.int)

# split test/train df
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

# model parameters
n_neurons_hidden = 128
n_neurons_out = label_names.size

# create model
model = tf.keras.Sequential([
	tf.keras.layers.Rescaling(1./255),
	tf.keras.layers.Normalization(axis=-1),
    tf.keras.layers.Dense(n_neurons_hidden, activation='relu'),
    tf.keras.layers.Dense(n_neurons_out, activation='softmax')
])

# compile model
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# train model
model.fit(x_train, y_train, epochs=10)

# evaluate accuracy
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)







# ------- IMPLEMENTATION 2 - FEED FORWARD NETWORK (only image data) ---------------------

# remove unused columns
df.pop('m_label')
df.pop('strength')
df.pop('italic')
df.pop('orientation')
df.pop('m_top')
df.pop('m_left')
df.pop('originalH')
df.pop('originalW')
df.pop('h')
df.pop('w')

# convert data to NP array
data = np.array(df)

# reshape image data
img_data = data.reshape(data.shape[0], 20, 20)

# split test/train df
x_train, x_test, y_train, y_test = train_test_split(img_data, labels, test_size=0.3)

# model parameters
n_neurons_hidden = 128
n_neurons_out = label_names.size

# create model
model = tf.keras.Sequential([
	tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Flatten(input_shape=(20, 20)),
    tf.keras.layers.Dense(n_neurons_hidden, activation='relu'),
    tf.keras.layers.Dense(n_neurons_out, activation='softmax')
])

# compile model
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# train model
model.fit(x_train, y_train, epochs=10)

# evaluate accuracy
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)





