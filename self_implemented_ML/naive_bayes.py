# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import *


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


# convert data to NP array
data = np.array(df)
data = data.astype(np.int)

# split test/train df
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)


# NAIVE BAYES IMPLEMENTATION
gaussians = dict()
prior_probabs= dict()
label_train = set(y_train)

for c in label_train:
  current_X = X_train[y_train==c]
  gaussians[c] = {
    'mean':current_X.mean(axis=0), # Mean for each font.
    'cov' : current_X.var(axis=0)+1e-2 # Covariance for each font with addition of a little noise to avoid singularity.
  }

  #prior probability is the number of times class occured divided by the sample length
  prior_probabs[c] = float(len([y_train==c]))/len(y_train)
  N, D = X_test.shape
  Probabilties = np.zeros((N, len(gaussians))) 

# Calculating the probabilities.
for c, g in gaussians.items():
  mean, cov = g['mean'], g['cov']
  Probabilties[:,c] = multivariate_normal.logpdf(X_test, mean=mean, cov=cov, allow_singular=True) + np.log(prior_probabs[c])
  
Prediction = np.argmax(Probabilties, axis=1)

