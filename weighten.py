#coding=utf-8
import xgboost as xgb
import lightgbm as lgb
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
label_train = train_df['is_trade'].values
data_train = train_df.drop(['instance_id', 'is_trade'], axis=1)
label_val = val_df['is_trade'].values
data_val = val_df.drop(['instance_id', 'is_trade'], axis=1)
id_test = test_df['instance_id']
data_test = test_df.drop(['instance_id'], axis=1)

p_xgb = 0.7
p_lgb = 0.3

model_xgb = joblib.load('model/xgb_model')
model_lgb = joblib.load('model/gbm')

def val():
    
    y_train_lgb = model_lgb.predict(data_train.values, num_iteration=model_lgb.best_iteration)
    dataset_train = xgb.DMatrix(data_train)
    y_train_xgb = model_xgb.predict(dataset_train, ntree_limit=model_xgb.best_iteration)
    
    y_val_lgb = model_lgb.predict(data_val.values, num_iteration=model_lgb.best_iteration)
    dataset_val = xgb.DMatrix(data_val)
    y_val_xgb = model_xgb.predict(dataset_val, ntree_limit=model_xgb.best_iteration)

    loss_train_lgb = log_loss(label_train, y_train_lgb)
    loss_val_lgb = log_loss(label_val, y_val_lgb)
    print 'lgb_loss: train: %f val: %f' %(loss_train_lgb, loss_val_lgb)
    
    loss_train_xgb = log_loss(label_train, y_train_xgb)
    loss_val_xgb = log_loss(label_val, y_val_xgb)
    print 'xgb_loss: train: %f val: %f' %(loss_train_xgb, loss_val_xgb)

    y_train = p_xgb * y_train_xgb + p_lgb * y_train_lgb
    y_val = p_xgb * y_val_xgb + p_lgb * y_val_lgb
    loss_train_weighten = log_loss(label_train, y_train)
    loss_val_weighten = log_loss(label_val, y_val)
    print 'weighten_loss: train: %f val: %f' %(loss_train_weighten, loss_val_weighten)

def submit():
    y_test_lgb = model_lgb.predict(data_test.values, num_iteration=model_lgb.best_iteration)
    dataset_test = xgb.DMatrix(data_test)
    y_test_xgb = model_xgb.predict(dataset_test, ntree_limit=model_xgb.best_iteration)

    y = p_xgb * y_test_xgb + p_lgb * y_test_lgb

    y = pd.Series(y, name='predicted_score')
    result = pd.concat([id_test, y], axis=1)
    time_format = '%Y-%m-%d-%H-%M-%S'
    time_now = datetime.datetime.now()
    bak_file = 'result/result_%s.csv'%time_now.strftime(time_format)
    result.to_csv(bak_file, index=False) # for backup
    result.to_csv('result/result.csv', index=False)
    print bak_file
    print y.mean()

def main():
    if len(sys.argv) == 2 and sys.argv[1] == 'submit':
        submit()
    else:
        val()

if __name__ == '__main__':
    main()