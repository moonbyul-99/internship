import lightgbm as lgb 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import pandas as pd 
from feature_generation import feature_generation 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score 
from feature_generation import feature_generation 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

def to_one_hot(y, num_classes=None):
    """
    将分类标签转换为one-hot编码形式。
    
    参数:
    y -- 一维整数数组，包含分类预测结果。
    num_classes -- 类别的总数。如果未提供，则从y中自动推断。
    
    返回:
    one_hot_matrix -- 二维数组，表示one-hot编码的预测矩阵。
    """
    if num_classes is None:
        num_classes = np.max(y) + 1
    one_hot_matrix = np.zeros((len(y), num_classes))
    one_hot_matrix[np.arange(len(y)), y] = 1
    return one_hot_matrix

def model_evaluation(gbm, dev_X, dev_y, verbose = 0):
    if verbose == 1:
        print(f'model evaluation stage ' + '===='*30)
    eval_result = {}
    res = gbm.predict(dev_X)
    pred_label = np.argmax(res, axis = 1)

    print(classification_report(dev_y, pred_label))

    C = np.unique(dev_y).shape[0]

    if verbose == 1:
        print(f"naive case in dev set {'===='*20}")
        for i in range(C):
            print(f'if pred all {i}, accuracy is {(dev_y == i).sum()/(dev_y.shape[0])}')

    dev_true = to_one_hot(dev_y, C)
    dev_pred = to_one_hot(pred_label, C)

    if verbose == 1:
        print('===='*20)
        print(f' acc score {accuracy_score(dev_y, pred_label)}')
        print(f' dev set predict label distribution: {dev_pred.sum(axis = 0)},  dev set true label distribution: {dev_true.sum(axis = 0)}')
        for i in range(C):
            print(f' acc score in class {i}: {accuracy_score(dev_true[:,i], dev_pred[:,i])}')

    eval_result['acc score'] = accuracy_score(dev_y, pred_label)
    for i in range(C):
        eval_result[f'acc_{i}'] = accuracy_score(dev_true[:,i], dev_pred[:,i])

    if verbose == 1:
        print(f" roc_auc score based on label prediction {'===='*20}")
        print(f" roc_auc score {roc_auc_score(dev_true, dev_pred, average=None)}")
    
    for i in range(C):
        eval_result[f'label_auc_{i}'] = roc_auc_score(dev_true, dev_pred, average=None)[i]

    if verbose == 1:
        print(f" roc_auc score based on score prediction {'===='*20}")

    pred_proba = gbm.predict(dev_X, raw_score=True)

    if verbose == 1:
        print(f" roc_auc score {roc_auc_score(dev_true, pred_proba, average=None)}")

    for i in range(C):
        eval_result[f'score_auc_{i}']= roc_auc_score(dev_true, pred_proba, average=None)[i]
    return pd.DataFrame(eval_result, index = ['eval_result'])

def lgb_loss_plot(record, title):
    plt.figure(figsize=(6, 4))
    plt.plot(list(record['training']['multi_logloss']), label='Training Loss')
    plt.plot(list(record['valid_1']['multi_logloss']), label='Valid Loss')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Binary LogLoss')
    plt.legend()
    plt.show()
    return None


def lgb_cv_eval(k,train_data, train_label, lgb_params, num_boost_round):
    #        lgb_params = {'learning_rate': 0.1, 'max_depth': -1, 'min_child_weight': 1,
    #                'colsample_bytree': 1, 'subsample': 1, 'reg_lambda': 0.5, 'reg_alpha': 0.5,'num_leaves':31,
    #                'seed': 33,'verbose':1,  'objective':'multiclass' , 'num_class': 3} 

    skf = StratifiedKFold(n_splits=k, shuffle = True, random_state=42)

    cv_model = {}
    cv_record = {}
    cv_dev = {}
    cv_train = {}
    for i, (train_idx, dev_idx) in enumerate(skf.split(train_data, train_label)):
        print('===='*15)
        print(f'fold {i}') 
        print('===='*15)
        train_X = train_data[train_idx,:]
        train_y = train_label[train_idx]

        dev_X = train_data[dev_idx,:]
        dev_y = train_label[dev_idx]
        
        lgb_train = lgb.Dataset(train_X, train_y)
        lgb_dev = lgb.Dataset(dev_X, dev_y, reference=lgb_train)

        record = {}
        gbm = lgb.train(lgb_params, lgb_train, num_boost_round= num_boost_round,valid_sets=[lgb_train, lgb_dev],
                        callbacks=[lgb.record_evaluation(record)])
        
        cv_record[i] = record
        cv_model[i] = gbm

        lgb_loss_plot(record, title = f'fold {i} training loss')

        eval_dev = model_evaluation(gbm, dev_X, dev_y)
        cv_dev[i] = eval_dev

        eval_train = model_evaluation(gbm, train_X, train_y)
        cv_train[i] = eval_train
        

    print(f'dev set evaluation result ' + '===='*15)
    dev_df = pd.DataFrame()
    for i in cv_dev.keys():
        tmp = cv_dev[i]
        tmp.index = [f'fold_{i}']
        dev_df = pd.concat([dev_df, tmp], axis= 0 )
    print(dev_df)

    print(f'train set evaluation result ' + '===='*15)
    train_df = pd.DataFrame()
    for i in cv_train.keys():
        tmp = cv_train[i]
        tmp.index = [f'fold_{i}']
        train_df = pd.concat([train_df, tmp], axis= 0 )
    print(train_df)
    return cv_model, dev_df, train_df


'''
def xgb_cv_eval(train_data, train_label, lgb_params, num_boost_round):
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state=42)

    cv_model = {}
    cv_record = {}
    cv_dev = {}
    cv_train = {}
    for i, (train_idx, dev_idx) in enumerate(skf.split(train_data, train_label)):
        print('===='*15)
        print(f'fold {i}') 
        print('===='*15)
        train_X = train_data[train_idx,:]
        train_y = train_label[train_idx]

        dev_X = train_data[dev_idx,:]
        dev_y = train_label[dev_idx]

        dtrain = xgb.DMatrix(train_X, label=train_y)
        dtest = xgb.DMatrix(dev_X, label=dev_y)

        # 设置参数
        params = {
            'eta': 0.1, 
            'max_depth': 6,  
            'objective': 'multi:softmax',  
            'eval_metric': 'mlogloss',
            'lambda': 1,
            'alpha': 0.1,
            'num_class':3
        }

        # 训练模型
        gbm = xgb.train(params, dtrain, num_boost_round=100)

        # 预测
        pred_label = gbm.predict(dtest)
        pred_score = gbm.predict(dtest, output_margin = True)  # 将概率转换为类别预测

        # 性能评估
        accuracy = accuracy_score(dev_y, pred_label)
        #auc_score = roc_auc_score(dev_y, pred_score, average=None)

        print(f"Accuracy: {accuracy:.4f}")
        #print(f"AUC Score: {auc_score:.4f}")
    return None 
'''

'''
def model_evaluation(gbm, dev_X, dev_y, verbose = 0):
    if verbose == 1:
        print(f'model evaluation stage ' + '===='*30)
    eval_result = {}
    res = gbm.predict(dev_X)
    pred_label = np.argmax(res, axis = 1)

    if verbose == 1:
        print(f"naive case in dev set {'===='*20}")
        print(f'if pred all zero, accuracy is {(dev_y == 0).sum()/(dev_y.shape[0])}')
        print(f'if pred all one, accuracy is {(dev_y == 1).sum()/(dev_y.shape[0])}')
        print(f'if pred all two, accuracy is {(dev_y == 2).sum()/(dev_y.shape[0])}')

    dev_true = to_one_hot(dev_y, 3)
    dev_pred = to_one_hot(pred_label, 3)

    if verbose == 1:
        print('===='*20)
        print(f' acc score {accuracy_score(dev_y, pred_label)}')
        print(f' dev set predict label distribution: {dev_pred.sum(axis = 0)},  dev set true label distribution: {dev_true.sum(axis = 0)}')
        print(f' acc score in class 0: {accuracy_score(dev_true[:,0], dev_pred[:,0])}')
        print(f' acc score in class 1: {accuracy_score(dev_true[:,1], dev_pred[:,1])}')
        print(f' acc score in class 2: {accuracy_score(dev_true[:,2], dev_pred[:,2])}')

    eval_result['acc score'] = accuracy_score(dev_y, pred_label)
    eval_result['acc_0'] = accuracy_score(dev_true[:,0], dev_pred[:,0])
    eval_result['acc_1'] = accuracy_score(dev_true[:,1], dev_pred[:,1])
    eval_result['acc_2'] = accuracy_score(dev_true[:,2], dev_pred[:,2])

    if verbose == 1:
        print(f" roc_auc score based on label prediction {'===='*20}")
        print(f" roc_auc score {roc_auc_score(dev_true, dev_pred, average=None)}")
    
    eval_result['label_auc_0'],eval_result['label_auc_1'], eval_result['label_auc_2'] = roc_auc_score(dev_true, dev_pred, average=None)

    if verbose == 1:
        print(f" roc_auc score based on score prediction {'===='*20}")

    pred_proba = gbm.predict(dev_X, raw_score=True)

    if verbose == 1:
        print(f" roc_auc score {roc_auc_score(dev_true, pred_proba, average=None)}")

    eval_result['score_auc_0'],eval_result['score_auc_1'], eval_result['score_auc_2'] = roc_auc_score(dev_true, pred_proba, average=None)
    return pd.DataFrame(eval_result, index = ['eval_result'])
'''