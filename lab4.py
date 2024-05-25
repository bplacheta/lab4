import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
data = pd.read_csv('C:\\Users\\blaze\\PycharmProjects\\lab4\\conv\\texture_features.csv')

if 'Kategoria' in data.columns:
    X = data.drop(columns=['Kategoria']).values
    y = data['Kategoria'].values
else:
    raise KeyError("Kolumna 'Kategoria' nie została znaleziona w pliku CSV.")

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, onehot_encoded, test_size=0.3)

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True)

y_pred = model.predict(X_test)
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test_int, y_pred_int)
print(cm)
