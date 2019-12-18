# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     xgblgb
   Description :
   Author :        Asdil
   date：          2018/10/31
-------------------------------------------------
   Change Activity:
                   2018/10/31:
-------------------------------------------------
"""
__author__ = 'Asdil'
import os
import time
import numpy as np
from Log import log
from Asdil import tool
from multiprocessing import Pool
from collections import Counter
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, anneal,tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

# LGB调参训练
def LGB(argsDict):
    num_leaves = argsDict["num_leaves"] + 25
    max_depth = argsDict["max_depth"]
    learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
    n_estimators = argsDict['n_estimators'] * 10 + 50
    min_child_weight = argsDict['min_child_weight']
    min_child_samples = argsDict['min_child_samples'] + 18
    subsample = argsDict["subsample"] * 0.1 + 0.7
    colsample_bytree = argsDict["colsample_bytree"]
    reg_alpha = argsDict["reg_alpha"]
    reg_lambda = argsDict["reg_lambda"]
    path = argsDict['path']
    data = np.load(path)
    data = data.astype('float32')
    data[data == 2] = 0.5
    X, Y = data[:, :-1], data[:, -1]
    _, rsid, _, _ = tool.splitPath(path)
    gbm = LGBMClassifier(
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        max_bin=255,
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=1
    )
    # kfold = StratifiedKFold(n_splits=5, random_state=42)
    kfold = StratifiedKFold(n_splits=5)
    metric = cross_val_score(gbm, X, Y, cv=kfold, scoring="roc_auc").mean()
    logger = log.init_log()
    logger.info(f"{rsid} 的训练得分为: {metric}")
    print(f"{rsid} 的训练得分为: {metric}")
    return -metric

def TRAINLGB(X, Y, argsDict, name, save_path, logger):
    num_leaves = argsDict["num_leaves"]
    max_depth = argsDict["max_depth"]
    learning_rate = argsDict["learning_rate"]
    n_estimators = argsDict['n_estimators']
    min_child_weight = argsDict['min_child_weight']
    min_child_samples = argsDict['min_child_samples']
    subsample = argsDict["subsample"]
    colsample_bytree = argsDict["colsample_bytree"]
    reg_alpha = argsDict["reg_alpha"]
    reg_lambda = argsDict["reg_lambda"]
    gbm = LGBMClassifier(device='gpu',
                         gpu_platform_id=0,
                         gpu_device_id=0,
                         max_bin=255,
                         num_leaves=num_leaves,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         min_child_weight=min_child_weight,
                         min_child_samples=min_child_samples,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         reg_alpha=reg_alpha,
                         reg_lambda=reg_lambda,
                         n_jobs=1
                         )
    #kfold = StratifiedKFold(n_splits=5, random_state=42)
    kfold = StratifiedKFold(n_splits=5)
    metric = cross_val_score(gbm, X, Y, cv=kfold, scoring="roc_auc").mean()
    logger.info(f"位点: {name} 最终得分: {metric}")
    print(f"位点: {name} 最终得分: {metric}")
    gbm.fit(X, Y)
    tail = '.'+str(int(round(metric, 5)*100000))
    joblib.dump(gbm, save_path + tail)

def RECOVERLGB(best): #因为返回的是索引，因此要还原
    best["num_leaves"] = best["num_leaves"] + 25   # randin
    best["max_depth"] = [-1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18][best["max_depth"]]  # choice
    best['learning_rate'] = best['learning_rate'] * 0.02 + 0.05    # uniform
    best['n_estimators'] = best['n_estimators'] * 10+50            # randin
    best['min_child_weight'] = best['min_child_weight']
    best['min_child_samples'] = best['min_child_samples'] + 18
    best['subsample'] = best['subsample'] * 0.1 + 0.7
    best['colsample_bytree'] = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0][best['colsample_bytree']]
    best['reg_alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1][best['reg_alpha']]
    best['reg_lambda'] = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100][best['reg_lambda']]
    return best

def pipeline(path):
    max_evals = 30
    _, name, _, _ = tool.splitPath(path)
    logger.info(f'开始训练位点: {name}')
    print(f'开始训练位点: {name}')

    data = np.load(path)

    try:
        X, Y = data[:, :-1], data[:, -1]
    except:
        logger.info(f'位点: {name} 文件读取错误')
        print(f'位点: {name} 文件读取错误')
        return 0

    if len(np.unique(Y)) == 1:
        logger.info(f'位点: {name} 只有一种类标签')
        print(f'位点: {name} 只有一种类标签')
        return 0

    tmp = Y.tolist()
    tmp = dict(Counter(tmp))
    if tmp[0] > tmp[1]:
        ma, mi = tmp[0], tmp[1]
    else:
        ma, mi = tmp[1], tmp[0]
    if mi / ma < 0.01:
        logger.info(f'位点: {name} 为低频位点')
        print(f'位点: {name} 为低频位点')
        return 0

    space = {
        "num_leaves": hp.randint("num_leaves", 5),  # [0, upper)
        "max_depth": hp.choice("max_depth", [-1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
        "learning_rate": hp.uniform("learning_rate", 0.001, 2),  # 0.001-2均匀分布
        "n_estimators": hp.randint("n_estimators", 5),  # [0,1000)
        "min_child_weight": hp.uniform("min_child_weight", 0.001, 0.01),  # 0.001-2均匀分布
        "min_child_samples": hp.randint("min_child_samples", 10),  # [0,1000)
        "subsample": hp.randint("subsample", 4),
        "colsample_bytree": hp.choice("colsample_bytree", [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
        "reg_alpha": hp.choice("reg_alpha", [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]),
        "reg_lambda": hp.choice("reg_lambda", [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]),
        "path": hp.choice('path', [path])
    }



    star = time.time()
    algo = partial(tpe.suggest, n_startup_jobs=1)  # 优化算法种类
    best = fmin(LGB, space, algo=algo, max_evals=max_evals)  # max_evals表示想要训练的最大模型数量，越大越容易找到最优解

    best = RECOVERLGB(best)
    TRAINLGB(X, Y, best, name, save_path + name + '.lgb', logger)
    end = time.time()
    times = end-star
    logger.info(f'位点: {name} 用时为: {times}')




if __name__ == '__main__':
    # 文件路径
    model_type = 'lgb'
    files_path = ''
    save_path = ''
    log.alter_log_ini()
    logger = log.init_log()
    logger.info('开始训练....')
    print('开始训练....')

    files = tool.getFiles(files_path, 'npy')
    logger.info('文件列表读取完成....')
    print('文件列表读取完成....')
    i = 0
    for path in files:
        pipeline(path)
        if i == 2:
            break
        i += 1

    logger.info('训练完毕....')
    print('训练完毕....')



