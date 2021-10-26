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

# create list of dfFrames
df_list = [pd.read_csv(path+file) for file in files]

# concatenate DFs together
df = pd.concat(df_list)

# remove unused columns
df.pop('fontVariant')
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
df.drop(df.index[df['font'] == 'scanned'], inplace=True)

# create X/y DFs
labels_df = df.pop('font')

# replace labels with integer values
labels_df_factorized, label_names = pd.factorize(labels_df)

# convert everything to NP arrays
data = np.array(df)
labels = np.array(labels_df_factorized)

# reshape image data
data = data.reshape(data.shape[0], 20, 20)

# split test/train df
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)


# IMPLEMENTATION 1 - FEED FORWARD NETWORK
# model parameters
n_neurons_hidden = 20
n_neurons_out = label_names.size

# create model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(20, 20)),
    tf.keras.layers.Dense(n_neurons_hidden, activation='sigmoid'),
    tf.keras.layers.Dense(n_neurons_out, activation='softmax')
])

# compile model
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train model
model.fit(x_train, y_train, epochs=10)

# evaluate accuracy
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)