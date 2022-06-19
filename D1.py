import numpy as np 
import pandas as pd 
import os

import matplotlib.pyplot as plt
import csv
import itertools
import collections

import pywt
from scipy import stats

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Conv1D, AvgPool1D, Flatten, Dense, Dropout, Softmax
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras import regularizers
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE


%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import LeakyReLU
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Add, Embedding, Conv1DTranspose, RepeatVector, Softmax, Conv1D, \
    Flatten, UpSampling1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import BatchNormalization

from keras.models import model_from_json
from sklearn.metrics import accuracy_score
import warnings
import glob
warnings.filterwarnings('ignore')



#########For mounting google drive###
from google.colab import drive
drive.mount('/content/gdrive')
files = [f for f in glob.glob("**/*.csv", recursive=True)]
print(files)
#######################################


FEATURES = 561

XTrain = np.loadtxt('/content/gdrive/MyDrive/X_train.csv', delimiter=',')
XTrain = np.reshape(XTrain, (XTrain.shape[0], FEATURES))
print(XTrain.shape)
n_inputs = XTrain.shape[1]

YTrain = np.loadtxt('/content/gdrive/MyDrive/y_train.csv', delimiter=',')
print(YTrain.shape)

XTest = np.loadtxt('/content/gdrive/MyDrive/X_test.csv', delimiter=',')
XTest = np.reshape(XTest, (XTest.shape[0], FEATURES))
print(XTest.shape)

YTest = np.loadtxt('/content/gdrive/MyDrive/y_test.csv', delimiter=',')
print(YTest.shape)


##For Training data

X = list()
X = [None] * len(XTrain)
for i in range(0,len(XTrain)):
        X[i] = np.append(XTrain[i], YTrain[i])

print(np.shape(X))

#Plotting Original data (Training)

X_train_df = pd.DataFrame(X)
per_class = X_train_df[X_train_df.shape[1]-1].value_counts()
print(per_class)
plt.figure(figsize=(20,15))
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(per_class, labels=['1', '2', '3', '4', '5','6', '7', '8', '9', '10','11','12'], colors=['tab:blue','tab:orange','tab:purple','tab:olive','tab:green','tab:grey','tab:blue','tab:pink','tab:red','tab:purple','tab:orange','tab:olive'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()



##For Testing data
Y = list()
Y = [None] * len(XTest)
for i in range(0,len(XTest)):
        Y[i] = np.append(XTest[i], YTest[i])

print(np.shape(Y))

#Plotting Original data (Testing)

Y_test_df = pd.DataFrame(Y)
per_class = Y_test_df[Y_test_df.shape[1]-1].value_counts()
print(per_class)
plt.figure(figsize=(20,15))
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(per_class, labels=['1', '2', '3', '4', '5','6', '7', '8', '9', '10','11','12'], colors=['tab:blue','tab:orange','tab:purple','tab:olive','tab:green','tab:grey','tab:blue','tab:pink','tab:red','tab:purple','tab:orange','tab:olive'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


## Resampling of classes

# For training data

df_1=X_train_df[X_train_df[X_train_df.shape[1]-1]==1]
df_2=X_train_df[X_train_df[X_train_df.shape[1]-1]==2]
df_3=X_train_df[X_train_df[X_train_df.shape[1]-1]==3]
df_4=X_train_df[X_train_df[X_train_df.shape[1]-1]==4]
df_5=X_train_df[X_train_df[X_train_df.shape[1]-1]==5]
df_6=X_train_df[X_train_df[X_train_df.shape[1]-1]==6]
df_7=X_train_df[X_train_df[X_train_df.shape[1]-1]==7]
df_8=X_train_df[X_train_df[X_train_df.shape[1]-1]==8]
df_9=X_train_df[X_train_df[X_train_df.shape[1]-1]==9]
df_10=X_train_df[X_train_df[X_train_df.shape[1]-1]==10]
df_11=X_train_df[X_train_df[X_train_df.shape[1]-1]==11]
df_12=X_train_df[X_train_df[X_train_df.shape[1]-1]==12]



df_1_upsample=resample(df_1,replace=True,n_samples=1000,random_state=122)
df_2_upsample=resample(df_2,replace=True,n_samples=1000,random_state=123)
df_3_upsample=resample(df_3,replace=True,n_samples=1000,random_state=124)
df_4_upsample=resample(df_4,replace=True,n_samples=1000,random_state=125)
df_5_upsample=resample(df_5,replace=True,n_samples=1000,random_state=126)
df_6_upsample=resample(df_6,replace=True,n_samples=1000,random_state=127)
df_7_upsample=resample(df_7,replace=True,n_samples=1000,random_state=128)
df_8_upsample=resample(df_8,replace=True,n_samples=1000,random_state=129)
df_9_upsample=resample(df_9,replace=True,n_samples=1000,random_state=130)
df_10_upsample=resample(df_10,replace=True,n_samples=1000,random_state=131)
df_11_upsample=resample(df_11,replace=True,n_samples=1000,random_state=132)
df_12_upsample=resample(df_12,replace=True,n_samples=1000,random_state=133)

# X_train_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample,df_5_upsample])
X_train_df=pd.concat([df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample,df_5_upsample,
                      df_6_upsample,df_7_upsample,df_8_upsample,df_9_upsample,
                      df_10_upsample,df_11_upsample,df_12_upsample])



per_class = X_train_df[X_train_df.shape[1]-1].value_counts()
print(per_class)
plt.figure(figsize=(20,15))
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(per_class, labels=['1', '2', '3', '4', '5','6', '7', '8', '9', '10','11','12'], colors=['tab:blue','tab:orange','tab:purple','tab:olive','tab:green','tab:grey','tab:blue','tab:pink','tab:red','tab:purple','tab:orange','tab:olive'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# Combining both dataframes
CL = pd.concat([X_train_df, Y_test_df], axis=0)

train, test = train_test_split(CL, test_size=0.25)

print("X_train : ", np.shape(train))
print("X_test  : ", np.shape(test))

# Train split
target_train = train[train.shape[1]-1]
print(np.shape(target_train))
target_test = test[test.shape[1]-1]
print(np.shape(target_test))
train_y = to_categorical(target_train)
test_y = to_categorical(target_test)
print(np.shape(train_y), np.shape(test_y))

# Test Split
train_x = train.iloc[:,:train.shape[1]-1].values
test_x = test.iloc[:,:test.shape[1]-1].values
train_x = train_x.reshape(len(train_x), train_x.shape[1],1)
test_x = test_x.reshape(len(test_x), test_x.shape[1],1)
print(np.shape(train_x), np.shape(test_x))

#LSTM

ac = tf.keras.layers.LeakyReLU(alpha=0.8)
model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.LSTM(units=10, activation='relu', input_shape=(n_inputs, 1)))
model1.add(Dense(13, activation=ac))
model1.add(tf.keras.layers.BatchNormalization())
model1.add(Dense(13, activation='softmax'))
model1.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model1.compile(optimizer=opt,
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
               
               
               #CNN

model3 = tf.keras.Sequential()
model3.add(tf.keras.Input(shape=(n_inputs, 1)))

model3.add(Conv1D(18, kernel_size=8, strides=1, padding='same', activation='relu'))
model3.add(MaxPooling1D(pool_size=1))

model3.add(Conv1D(32, kernel_size=18, strides=1, padding='same', activation='relu'))
model3.add(MaxPooling1D(pool_size=2))

model3.add(Conv1D(50, kernel_size=32, strides=2, padding='same', activation='relu'))
model3.add(MaxPooling1D(pool_size=1))

model3.add(tf.keras.layers.Flatten(data_format=None))
model3.add(tf.keras.layers.BatchNormalization())
model3.add(Dense(13, activation='relu'))
model3.add(Dense(13, activation='softmax'))
model3.summary()

model3.compile(optimizer=opt,
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
               
               
               ############ Ensembling all models

mergedOut = Add()([model1.output, model3.output])
newModel = Model([model1.input, model3.input], mergedOut)
newModel.summary()

newModel.compile(optimizer=opt,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

newModel.fit([train_x, train_x], train_y, epochs=5, batch_size=200, shuffle=False)

preddd = newModel.predict([test_x, test_x])
print(preddd.shape)
preddd = np.argmax(preddd, axis=1)

# #   Ensembler Accuracy
print(preddd)


test_y = np.argmax(test_y, axis=1)
Accuracy = accuracy_score(test_y, preddd)
NAccuracy = Accuracy * 100
print('Merged Model Accuracy:', NAccuracy, '%')

#####   Generating confusion matrix of HEARTBEATS

import sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix

class_labels = ['1', '2', '3', '4', '5','6', '7', '8', '9', '10','11', '12']
confusion_matrix = confusion_matrix(test_y, preddd)
sns.heatmap(confusion_matrix, xticklabels = class_labels, yticklabels = class_labels, annot = True, linewidths = 0.1, fmt='d', cmap = 'YlGnBu')
plt.title("Confusion matrix", fontsize = 15)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
