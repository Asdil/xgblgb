earch
'''n_estimators 默认值100     含义: 总共迭代的次数，即决策树的个数
   max_depth    默认值6       含义: 树的深度，典型值3-10
   min_child_weight 默认值1   含义: 值越大，越容易欠拟合；值越小，越容易过拟合（值较大时，避免模型学习到局部的特殊样本）
   subsample 默认值1          含义: 训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1 防止overfiting
   colsample_bytree 默认值1   含义：训练每棵树时，使用的特征占全部特征的比例。典型值为0.5-1  防止overfiting
   '''
'''learning_rate 默认值0.1    含义: 学习率，控制每次迭代更新权重时的步长
    objective 目标函数   回归任务
                                reg:linear (默认)
                                reg:logistic 
                       二分类
                                binary:logistic     概率 
                                binary：logitraw   类别
                       多分类
                                multi：softmax  num_class=n   返回类别
                                multi：softprob   num_class=n  返回概率
                                rank:pairwise 
    eval_metric 评价参数
                        回归任务(默认rmse)
                                rmse--均方根误差
                                mae--平均绝对误差
                        分类任务(默认error)
                                auc--roc曲线下面积
                                error--错误率（二分类）
                                merror--错误率（多分类）
                                logloss--负对数似然函数（二分类）
                                 mlogloss--负对数似然函数（多分类）
    gamma 默认值0   含义: 惩罚项系数，指定节点分裂所需的最小损失函数下降值
    alpha 默认值1   含义: L1正则化系数
    lambda 默认值1  含义: L2正则化系数'''
import numpy as np
import xgboost as xgb
from sklearn.grid_search import RandomizedSearchCV
clf = xgb.XGBClassifier()
param_dist = {
        'n_estimators':range(100,400,4),
        'max_depth':range(2,15,1),
        'learning_rate':np.linspace(0.01,2,20),
        'subsample':np.linspace(0.7,0.9,20),
        'colsample_bytree':np.linspace(0.5,0.98,10),
        'min_child_weight':range(1,9,1),
        'gamma':[i/10.0 for i in range(0,5)]，
        'alpha': [0, 0.25, 0.5, 0.75, 1],
        'lambda': [0, 0.2, 0.4, 0.6, 0.8, 1],
        }
grid = RandomizedSearchCV(estimator=clf,
                          param_distributions=param_dist,
                          cv=5,
                          scoring='neg_log_loss',
                          n_iter=300,
                          n_jobs=1)
# grid.grid_scores_, grid.best_params_, grid.best_score_




# lightgbm random_search
import numpy as np
import lightgbm as lgb
from sklearn.grid_search import RandomizedSearchCV
'''boosting_type 树模型种类 default=gbdt    
                options=gbdt 传统的梯度提升决策树, 
                rf           随机森林, 
                dart         Dropouts meet Multiple Additive Regression Trees, 
                goss         Gradient-based One-Side Sampling基于梯度的单侧采样
                
   n_jobs      线程数      default=-1   这里官方文档提到，数字设置成cpu内核数比线程数训练效更快(考虑到现在cpu大多超线程)。并行学习不应该设置成全部线程，这反而使得训练速度不佳。
                
                
   learning_rate 学习速率   default=0.1
   
                
    num_leaves  叶子数       default=31       Lightgbm用决策树叶子节点数来确定树的复杂度
    
    device      设备   default=cpu, options=cpu, gpu
    max_depth   default=-1, 限制树模型的最大深度. 这可以在的情况下防止过拟合. 
                           树仍然可以通过 leaf-wise 生长.
    min_child_samples   default=20 一个叶子上数据的最小数量. 可以用来处理过拟合 
    min_child_weight    default=1e-3 一个叶子上的最小 hessian 和. 类似于min_child_samples
                        子节点所需的样本权重和(hessian)的最小阈值，
                        若是基学习器切分后得到的叶节点中样本权重和低于该阈值则不会进一步切分，
                        在线性模型中该值就对应每个节点的最小样本数，
                        该值越大模型的学习约保守，同样用于防止模型过拟合
                        
    colsample_bytree default=1.0   LightGBM 将会在每次迭代中随机选择部分特征. 例如, 如果设置为 0.8, 
                                   将会在每棵树训练之前选择80%的特征可以用来加速训练可以用来处理过拟合
    subsample   default=1.0   但是它将在不进行重采样的情况下随机选择部分数据可以用来
                              加速训练可以用来处理过拟合
    min_split_gain   default=0    执行切分的最小增益
    
    
     
    '''


param_dist = {
        'num_leaves':range(25, 36),
        'max_depth': [-1, 15, 20, 25, 30, 35],
        'learning_rate':np.linspace(0.01,2,20),
        'n_estimators': range(100,400,4),
        'min_child_weight':np.linspace(0.001, 0.01, 10),
        'min_child_samples':[15,20,25],
        'subsample':np.linspace(0.8, 1.0, 5),
        'colsample_bytree':np.linspace(0.8, 1.0, 5),
        'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
        'reg_lambda': [0, 0.2, 0.4, 0.6, 0.8, 1],
        }

clf = lgb.LGBMClassifier(objective='regression',
                        num_leaves=31,
                        max_bin = 300,
                        learning_rate=0.05,
                        n_estimators=100,
                        n_jobs=8)
grid = RandomizedSearchCV(estimator=clf,
                          param_distributions=param_dist,
                          cv=5,
                          scoring='neg_log_loss',
                          n_iter=300,
                          n_jobs=1)
# grid.grid_scores_, grid.best_params_, grid.best_score_
