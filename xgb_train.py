# coding=utf-8
# pylint:disable=E1101
import xgboost as xgb
import pandas as pd
import numpy as np
import datetime
import sys

from sklearn.externals import joblib

def log_loss(y_list, p_list):
    assert len(y_list) == len(p_list), 'length not match {} vs. {}'.format(
        len(y_list), len(p_list))
    n = len(y_list)
    ans = 0
    for i in xrange(n):
        yi = y_list[i]
        pi = p_list[i]
        ans += yi * np.log(pi) + (1 - yi) * np.log(1 - pi)
    ans = -ans / n
    return ans

train_df = pd.read_csv('data/train_feat.csv')
val_df = pd.read_csv('data/val_feat.csv')
test_df = pd.read_csv('data/test_feat.csv')

train_df['context_page_id_1'] = (train_df['context_page_id'] <= 1)
test_df['context_page_id_1'] = (test_df['context_page_id'] <= 1)
val_df['context_page_id_1'] = (val_df['context_page_id'] <= 1)

train_df['context_page_id_4'] = (train_df['context_page_id'] <= 4)
test_df['context_page_id_4'] = (test_df['context_page_id'] <= 4)
val_df['context_page_id_4'] = (val_df['context_page_id'] <= 4)

train_df['context_page_id_8'] = (train_df['context_page_id'] <= 8)
test_df['context_page_id_8'] = (test_df['context_page_id'] <= 8)
val_df['context_page_id_8'] = (val_df['context_page_id'] <= 8)

label_train = train_df['is_trade']
data_train = train_df.drop(['instance_id', 'is_trade'], axis=1)
label_val = val_df['is_trade']
data_val = val_df.drop(['instance_id', 'is_trade'], axis=1)
id_test = test_df['instance_id']
data_test = test_df.drop(['instance_id'], axis=1)
dataset_train = xgb.DMatrix(data_train, label=label_train)
dataset_val = xgb.DMatrix(data_val, label=label_val)
dataset_test = xgb.DMatrix(data_test)

params = {'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'gamma': 0.1,
            'min_child_weight': 3,  # 调大可以控制过拟合
            'max_depth': 6,
            'lambda': 10,
            'subsample': 0.5,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.7,
            'eta': 0.02,
            'tree_method': 'exact',
            'seed': 0,
            'nthread': -1
}



def train():
    
    watchlist = [(dataset_train, 'train'), (dataset_val, 'val')]
    model = xgb.train(params, dataset_train, num_boost_round=3000, evals=watchlist, early_stopping_rounds=200)


    y_pred = model.predict(dataset_train, ntree_limit=model.best_iteration)
    train_loss = log_loss(label_train, y_pred)

    y_pred = model.predict(dataset_val, ntree_limit=model.best_iteration)
    val_loss = log_loss(label_val, y_pred)
    
    joblib.dump(model, 'model/xgb_model')

    print 'train_loss: {:4}, val_loss: {:4}'.format(train_loss, val_loss)
    with open('feature_importance_xgb.txt','w') as f:
        feature_score = model.get_fscore()
        feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
        fs = []
        print '{} features'.format(len(feature_score))
        for (key,value) in feature_score:
            fs.append("{0},{1}\n".format(key,value))
        f.writelines(fs)

def submit():
    
    model = joblib.load('model/xgb_model')
    y = model.predict(dataset_test, ntree_limit=model.best_iteration)
    y = pd.Series(y.T, name='predicted_score')
    result = pd.concat([id_test, y], axis=1)
    time_format = '%Y-%m-%d-%H-%M-%S'
    time_now = datetime.datetime.now()
    bak_file = 'result/result_%s.csv'%time_now.strftime(time_format)
    result.to_csv(bak_file, index=False, sep=' ', mode='wb') # for backup
    result.to_csv('result/result.csv', index=False, sep=' ', mode='wb')
    print bak_file
    print y.mean()


def main():
    if len(sys.argv) == 2 and sys.argv[1] == 'submit':
        submit()
    else:
        train()

if __name__ == '__main__':
    main()