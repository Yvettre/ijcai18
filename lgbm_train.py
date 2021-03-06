# coding=utf-8
# pylint:disable=E1101
import lightgbm as lgb
import pandas as pd
import numpy as np
import datetime
import sys
import time

from sklearn.cross_validation import KFold
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

train_df = pd.read_csv('data/qing/train_feat.csv')
val_df = pd.read_csv('data/qing/val_feat.csv')
test_df = pd.read_csv('data/qing/test_feat.csv')

# train_df['time_trade_rate_x_today_cate_query_hour'] = train_df['time_trade_rate'] * train_df['today_cate_query_hour']
# val_df['time_trade_rate_x_today_cate_query_hour'] = val_df['time_trade_rate'] * val_df['today_cate_query_hour']
# test_df['time_trade_rate_x_today_cate_query_hour'] = test_df['time_trade_rate'] * test_df['today_cate_query_hour']

train_df['context_page_id_1'] = (train_df['context_page_id'] <= 1)
test_df['context_page_id_1'] = (test_df['context_page_id'] <= 1)
val_df['context_page_id_1'] = (val_df['context_page_id'] <= 1)

train_df['context_page_id_4'] = (train_df['context_page_id'] <= 4)
test_df['context_page_id_4'] = (test_df['context_page_id'] <= 4)
val_df['context_page_id_4'] = (val_df['context_page_id'] <= 4)

train_df['context_page_id_8'] = (train_df['context_page_id'] <= 8)
test_df['context_page_id_8'] = (test_df['context_page_id'] <= 8)
val_df['context_page_id_8'] = (val_df['context_page_id'] <= 8)

# drop_list = ['time_trade_num','time_not_trade_num','time_view_num','time_trade_rate']
# train_df[drop_list] = 0
# test_df[drop_list] = 0
# val_df[drop_list] = 0

label_train = train_df['is_trade'].values
data_train = train_df.drop(['instance_id', 'is_trade'], axis=1).values
label_val = val_df['is_trade'].values
data_val = val_df.drop(['instance_id', 'is_trade'], axis=1).values
id_test = test_df['instance_id']
data_test = test_df.drop(['instance_id'], axis=1).values

params = {
    'task': 'train',
    'boosting': 'gbdt',
    'objective': 'binary',
    'learning_rate': 0.017,
    'min_split_gain': 1.0,
    'metric': {'binary_logloss'},
    'metric_freq': 1,
    'is_training_metric': False,
    'num_leaves': 64,
    'feature_fraction': 0.35,
    'feature_fraction_seed': 6,   
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': -1,
    'min_data_in_leaf': 2000,
    'max_depth':6,
    'min_sum_hessian_in_leaf': 120,
}

def train():
    time_node1 = time.clock()

    feature_name = list(train_df.drop(['instance_id', 'is_trade'], axis=1).columns)
    print feature_name

    print data_train.shape, label_train.shape
    print data_val.shape, label_val.shape

    lgb_train = lgb.Dataset(data_train, label_train)
    lgb_eval = lgb.Dataset(data_val, label_val, reference=lgb_train)

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=lgb_eval,
        feature_name=feature_name,
        categorical_feature=[
            'user_gender_id', 'user_occupation_id', 'predict_major_cate', 'item_city_id', 'item_major_cate','item_second_cate'
        ],
        early_stopping_rounds=200)

    y_pred = gbm.predict(data_train, num_iteration=gbm.best_iteration)
    train_loss = log_loss(label_train, y_pred)

    y_pred = gbm.predict(data_val, num_iteration=gbm.best_iteration)
    val_loss = log_loss(label_val, y_pred)

    joblib.dump(gbm, 'model/qing/gbm')
    print 'train_loss: {:4}, val_loss: {:4}'.format(train_loss, val_loss)
    with open('feature_importance.txt','w') as f:
        importance = gbm.feature_importance()
        names = gbm.feature_name()
        pair = zip(names, importance)
        pair.sort(key=lambda x:x[1], reverse=True)
        print '{} features'.format(len(names))
        print pair
        for item in pair:
            string = item[0] + ', ' + str(item[1]) + '\n'
            f.write(string)
    
    time_node2 = time.clock()
    time_counter_tmp = time_node2 - time_node1
    print 'lgbm_train.py cost whole time: %dh-%dm-%ds'%(time_counter_tmp/3600, (time_counter_tmp%3600)/60, (time_counter_tmp%3600)%60)

    
def submit():
    gbm = joblib.load('model/qing/gbm')

    y = gbm.predict(data_test, num_iteration=gbm.best_iteration)

    y = pd.Series(y, name='predicted_score')
    result = pd.concat([id_test, y], axis=1)
    time_format = '%Y-%m-%d-%H-%M-%S'
    time_now = datetime.datetime.now()
    bak_file = 'result/result_%s.txt'%time_now.strftime(time_format)
    result.to_csv(bak_file, index=False, sep=' ', mode='wb') # for backup
    # result.to_csv('result/result.csv', index=False, sep=' ', mode='wb')
    print bak_file
    print y.mean()

def debug():
    gbm = joblib.load('model/qing/gbm')
    id_val = val_df['instance_id']
    y = gbm.predict(data_val, num_iteration=gbm.best_iteration)
    assert len(y) == len(label_val), 'length not match: {} vs. {}'.format(len(y), len(label_val))
    result = pd.DataFrame({'instance_id':id_val,'is_trade':label_val,'predicted_score':y})
    result['diff'] = np.abs(result['is_trade'] - result['predicted_score'])
    result.sort_values(by='diff', ascending=False, inplace=True)
    result.to_csv('debug/result_debug.csv', index=False)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == 'submit':
        submit()
    elif len(sys.argv) == 2 and sys.argv[1] == 'debug':
        debug()
    else:
        train()

if __name__ == '__main__':
    main()