# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     xgbmuti
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


# 超参数调节
def XGB(argsDict):
    max_depth = argsDict["max_depth"] + 1
    n_estimators = argsDict['n_estimators'] * 10+50
    learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
    subsample = argsDict["subsample"] * 0.1 + 0.7
    min_child_weight = argsDict["min_child_weight"]+1
    reg_alpha = argsDict["reg_alpha"]
    reg_lambda = argsDict["reg_lambda"]
    colsample_bytree = argsDict["colsample_bytree"]

    path = argsDict['path']
    data = np.load(path)
    data = data.astype('float32')
    data[data == 2] = 0.5
    X, Y = data[:, :-1], data[:, -1]
    _, rsid, _, _ = tool.splitPath(path)

    gbm = XGBClassifier(tree_method='gpu_hist',
                        max_bin=255,
                        objective="binary:logistic",
                        max_depth=max_depth,  #最大深度
                        n_estimators=n_estimators,   #树的数量
                        learning_rate=learning_rate, #学习率
                        subsample=subsample,      #采样数
                        min_child_weight=min_child_weight,   #孩子数
                        max_delta_step=10,  #10步不降则停止
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        colsample_bytree=colsample_bytree,
                       )
    kfold = StratifiedKFold(n_splits=5, random_state=42)
    metric = cross_val_score(gbm, X, Y, cv=kfold, scoring="roc_auc").mean()

    logger = log.init_log()
    logger.info(f"{rsid} xgb的训练得分为: {metric}")
    print(f"{rsid} xgb的训练得分为: {metric}")
    return -metric

# 得到最终参数后，训练模型
def TRAINXGB(X, Y, argsDict, name, save_path):
    logger = log.init_log()
    max_depth = argsDict["max_depth"]
    n_estimators = argsDict['n_estimators']
    learning_rate = argsDict["learning_rate"]
    subsample = argsDict["subsample"]
    min_child_weight = argsDict["min_child_weight"]
    reg_alpha = argsDict["reg_alpha"]
    reg_lambda = argsDict["reg_lambda"]
    colsample_bytree = argsDict["colsample_bytree"]
    gbm = XGBClassifier(tree_method='gpu_hist',
                        max_bin=800,
                        objective="binary:logistic",
                        n_jobs=16,
                        max_depth=max_depth,  #最大深度
                        n_estimators=n_estimators,   #树的数量
                        learning_rate=learning_rate, #学习率
                        subsample=subsample,      #采样数
                        min_child_weight=min_child_weight,   #孩子数
                        max_delta_step=10,  #10步不降则停止
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        colsample_bytree=colsample_bytree,
                       )
    kfold = StratifiedKFold(n_splits=5, random_state=42)
    metric = cross_val_score(gbm, X, Y, cv=kfold, scoring="roc_auc").mean()

    logger.info(f"位点: {name} xgb最终得分: {metric}")
    print(f"位点: {name} xgb最终得分: {metric}")
    gbm.fit(X, Y)
    tail = '.'+str(int(round(metric, 5)*100000))
    joblib.dump(gbm, save_path + tail)


def RECOVERXGB(best): #因为返回的是索引，因此要还原
    best["max_depth"] = best["max_depth"] + 1
    best['n_estimators'] = best['n_estimators'] * 10 + 50
    best["learning_rate"] = best["learning_rate"] * 0.02 + 0.05
    best["subsample"] = best["subsample"] * 0.1 + 0.7
    best["min_child_weight"] = best["min_child_weight"] + 1
    best["colsample_bytree"] = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0][best["colsample_bytree"]]
    best["reg_alpha"] = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1][best["reg_alpha"]]
    best["reg_lambda"] = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100][best["reg_lambda"]]
    return best


def pipeline(path):
    logger = log.init_log()
    max_evals = 50

    _, name, _, _ = tool.splitPath(path)
    logger.info(f'xgb开始训练位点: {name}')
    print(f'xgb开始训练位点: {name}')

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
        "max_depth": hp.randint("max_depth", 15),  # [0, upper)
        "n_estimators": hp.randint("n_estimators", 5),  # [0,1000)
        "learning_rate": hp.uniform("learning_rate", 0.001, 2),  # 0.001-2均匀分布
        "min_child_weight": hp.randint("min_child_weight", 5),
        "subsample": hp.randint("subsample", 4),
        "reg_alpha": hp.choice("reg_alpha", [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]),
        "reg_lambda": hp.choice("reg_lambda", [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]),
        "colsample_bytree": hp.choice("colsample_bytree", [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
        "path": hp.choice('path', [path])
        }

    star = time.time()
    algo = partial(tpe.suggest, n_startup_jobs=1)  # 优化算法种类
    best = fmin(XGB, space, algo=algo, max_evals=max_evals)  # max_evals表示想要训练的最大模型数量，越大越容易找到最优解

    best = RECOVERXGB(best)
    print(best)
    TRAINXGB(X, Y, best, name, save_path + name + '.xgb')
    end = time.time()
    times = end-star
    logger.info(f'位点: {name} xgb用时为: {times}')


if __name__ == '__main__':
    global save_path

    # 文件路径
    model_type = 'xgb'
    files_path = ''
    save_path = ''
    log.alter_log_ini()
    logger = log.init_log()
    logger.info('开始训练....')
    print('开始训练....')

    files = tool.getFiles(files_path, 'npy')
    logger.info('文件列表读取完成....')
    print('文件列表读取完成....')

#    pool = Pool(8)
#    pool.map(pipeline, files)
#    pool.close()
#    pool.join()


    i = 0
    for path in files:
        pipeline(path)
        if i == 1:
            break
        i += 1

    logger.info('训练完毕....')
    print('训练完毕....')
