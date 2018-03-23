#coding=utf-8
#pylint:disable=E1101
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

train_df['context_page_id_1'] = (train_df['context_page_id'] <= 1)
test_df['context_page_id_1'] = (test_df['context_page_id'] <= 1)
val_df['context_page_id_1'] = (val_df['context_page_id'] <= 1)

train_df['context_page_id_4'] = (train_df['context_page_id'] <= 4)
test_df['context_page_id_4'] = (test_df['context_page_id'] <= 4)
val_df['context_page_id_4'] = (val_df['context_page_id'] <= 4)

train_df['context_page_id_8'] = (train_df['context_page_id'] <= 8)
test_df['context_page_id_8'] = (test_df['context_page_id'] <= 8)
val_df['context_page_id_8'] = (val_df['context_page_id'] <= 8)

label_train = train_df['is_trade'].values
data_train = train_df.drop(['instance_id', 'is_trade'], axis=1)
label_val = val_df['is_trade'].values
data_val = val_df.drop(['instance_id', 'is_trade'], axis=1)
id_test = test_df['instance_id']
data_test = test_df.drop(['instance_id'], axis=1)

p_xgb = 0.55
p_lgb = 0.45

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

    min_val_loss = 1
    best_p_xgb = 0
    best_p_lgb = 1
    for i in range(0, 21):
        p_xgb = i / 20.0
        p_lgb = 1 - p_xgb
        y_train = p_xgb * y_train_xgb + p_lgb * y_train_lgb
        y_val = p_xgb * y_val_xgb + p_lgb * y_val_lgb
        loss_train_weighten = log_loss(label_train, y_train)
        loss_val_weighten = log_loss(label_val, y_val)
        print 'p_xgb: %f, p_lgb: %f, loss_train: %f, loss_val: %f'%(p_xgb, p_lgb, loss_train_weighten, loss_val_weighten)
        if loss_val_weighten < min_val_loss:
            min_val_loss = loss_val_weighten
            min_train_loss = loss_train_weighten
            best_p_lgb = p_lgb
            best_p_xgb = p_xgb
    print 'best p_xgb: %f best p_lgb: %f'%(best_p_xgb, best_p_lgb)
    print 'weighten_loss: train: %f val: %f' %(min_train_loss, min_val_loss)

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
    result.to_csv(bak_file, index=False, sep=' ', mode='wb') # for backup
    result.to_csv('result/result.csv', index=False, sep=' ', mode='wb')
    print bak_file
    print y.mean()

def main():
    if len(sys.argv) == 2 and sys.argv[1] == 'submit':
        submit()
    else:
        val()

if __name__ == '__main__':
    main()