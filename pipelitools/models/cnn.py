import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Input, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization

from tensorflow.keras import optimizers

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint, LambdaCallback

import pipelitools as t
from pipelitools.models import metrics as mt


tf.random.set_seed(42)

import os

for dirname, _, filenames in os.walk('../data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# mitbih data
df_train = pd.read_csv('./data/mitbih_train.csv', header=None)
df_test = pd.read_csv('./data/mitbih_test.csv', header=None)

# combined df
train = df_train.rename(columns={187: 'y'})
test = df_test.rename(columns={187: 'y'})

# training data
X_train = train[train.columns[:-1]]
y_train = train[train.columns[-1]]

# testing data
X_test = test[test.columns[:-1]]
y_test = test[test.columns[-1]]

#train validation split
X_train, X_val, y_train, y_val = train_test_split(train.iloc[:,:-1], train.iloc[:,-1],
                                                    test_size=0.2, random_state=42)

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

y_train_dummy=to_categorical(y_train)
y_val_dummy=to_categorical(y_val)
y_test_dummy=to_categorical(y_test)
y_train_dummy.shape

# For conv1D dimentionality should be 187X1 where 187 is number of features and 1 = 1D Dimentionality of data
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_val = X_val.reshape(len(X_val), X_val.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)


tf.random.set_seed(42)


def create_model(kernel_size=6, padding='same', strides=2, pool_size=2, lr=0.001, cl=2, cf=64, dl=2, dn=64, dense=True):
    """
    cl - CNN layers
    cf - CNN filters
    dl - DNN layers
    dn - DNN neurons
    """
    model = Sequential()

    model.add(Conv1D(filters=cf, kernel_size=kernel_size, activation='relu', padding=padding,
                     input_shape=(X_train.shape[1], 1)))
    model.add(BatchNormalization())  # Normalization to avoid overfitting
    model.add(MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding))

    # Add as many hidden layers as specified in nl
    for i in range(cl):
        # Layers have nn filters
        model.add(Conv1D(filters=cf, kernel_size=kernel_size, activation='relu', padding=padding))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding))

    model.add(Flatten())

    if dense:
        for i in range(dl):
            model.add(Dense(units=dn, activation='relu'))

    # Output Layer
    model.add(Dense(5, activation='softmax'))  # output layer

    # loss = 'categorical_crossentropy'
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


model = KerasClassifier(build_fn=create_model, verbose=1)

# Define a series of parameters. Keys in the param dict mus be named EXACTLY as the params in create_model function
params = {
    #     'activation': ['relu','softmax'],
    'kernel_size': [3, 5, 7],
    'padding': ['same'],
    'strides': [1, 2],
    'pool_size': [2, 5],
    'cl': [2, 5, 10],
    'cf': [32, 64, 128],
    'dl': [1, 2],
    'dn': [32, 64, 128],
    'dense': [True, False]
}

# Create a random search cv object and fit it to the data
# cv = RandomizedSearchCV(
#     estimator=model,
#     param_distributions=params,
#     cv=5,
#     random_state=42,
#     verbose=3,
# #     n_iter=10,
# #     n_jobs=10,
# )
cv = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    verbose=3,
    #     n_iter=10,
    n_jobs=-1,
)

name = 'CNN'

early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=5)

history = History()

mc = ModelCheckpoint(f"./temp_pickle_models/{name}.h5", monitor='val_accuracy', save_best_only=True)
# # load a saved model
# from keras.models import load_model
# saved_model = load_model('best_model.h5')


cv_results = cv.fit(X_train, y_train_dummy,
                    batch_size=32,
                    epochs=15,
                    validation_data=(X_val, y_val_dummy),
                    callbacks=[early_stopping_monitor, mc, history])

print(cv.best_params_)

y_pred = cv.predict(X_test)

name = 'CNN'

y_test_dummy = np.array(pd.get_dummies(y_test))
y_val_dummy = np.array(pd.get_dummies(y_val))
y_train_dummy = np.array(pd.get_dummies(y_train))

roc = mt.ROCcurve_multiclass
pr = mt.PR_multiclass
cm = mt.CM

mt.compare_models(cv, name, X_test, y_test_dummy, y_train_dummy, cm, roc, pr, proba=True, data='test')


















