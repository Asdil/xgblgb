# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     sss
   Description :
   Author :        Asdil
   date：          2018/9/10
-------------------------------------------------
   Change Activity:
                   2018/9/10:
-------------------------------------------------
"""
__author__ = 'Asdil'
import os
from Log import log
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))

from keras.optimizers import SGD,Adagrad
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
import keras
import time

global input_dim


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def create_model():
    model = Sequential()
    model.add(Dense(units=500, input_dim=input_dim, kernel_initializer='lecun_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=200, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adamax')
    return model


def gridsearch(X,Y, model, _file, save_path, n_splits=5):
    logger = log.init_log()
    ret = []
    skf = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index,:], X[test_index,:]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model.fit(X_train, Y_train, epochs=30, batch_size=64)
        score = roc_auc_score(Y_test, model.predict(X_test))
        print(model.predict(X_test))
        ret.append(score)
        logger.info(_file + ": " + str(score))
        print(_file + ": " + str(score))
    score = np.mean(ret)
    logger.info(_file + "平均auc: " + str(score))
    logger.info("开始训练全量模型" + str(score))
    print(_file + "平均auc: " + str(score))
    model.fit(X, Y, epochs=30, batch_size=64)
    model.save(save_path)



path = '/root/jiapeiling/v2-v2data/'
save_path = '/root/jiapeiling/v2-v2model/'
files = [path+each for each in os.listdir(path) if '.csv' in each]
log.alter_log_ini()
logger = log.init_log()
for _file in files:
    print(_file)
    file_name = _file.split('/')[-1].split('.')[0]
    df = pd.read_csv(_file)
    print(df.shape)
    X, Y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values
    input_dim = X.shape[1]
    print(X.shape[1])
    star = time.time()
    model = create_model()
    gridsearch(X, Y, model, _file, save_path + file_name + '.h5')
    end = time.time()
    logger.info('用时为: ' + str(end - star))
    
