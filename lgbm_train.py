# coding=utf-8
import lightgbm as lgb
import pandas as pd
import numpy as np
import datetime

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

train_df = pd.read_csv('data/train_feat.csv')
val_df = pd.read_csv('data/val_feat.csv')
test_df = pd.read_csv('data/test_feat.csv')
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
    # 'num_class': 2,
    # 'num_leaves': 35,
    # 'max_depth':10,
    'learning_rate': 0.02,
    # 'n_estimators':100,
    'min_split_gain': 1.0,
    'min_child_weight': 1.2,
    # 'colsample_bytree': 0.9,
    'metric': {'binary_logloss'},
    'metric_freq': 1,
    # 'is_training_metric': True,
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.9,
    # 'bagging_freq': 5,
    'verbose': -1,
    'min_data_in_leaf': 5
}

def cv():
    K = 5
    kf = KFold(train_df.shape[0], K, shuffle=True, random_state=0)
    train_loss_list = []
    val_loss_list = []
    feature_name = list(train_df.columns[1:-1])
    print feature_name
    i = 0
    for train_index, val_index in kf:
        i += 1
        x_train, x_val = data_train[train_index], data_train[val_index]
        y_train, y_val = label_train[train_index], label_train[val_index]

        print x_train.shape, y_train.shape
        print x_val.shape, y_val.shape

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=lgb_eval,
            feature_name=feature_name,
            categorical_feature=[
                'user_gender_id', 'user_occupation_id'
            ],
            early_stopping_rounds=20)

        y_pred = gbm.predict(x_val, num_iteration=gbm.best_iteration)
        train_loss = log_loss(y_val, y_pred)
        train_loss_list.append(train_loss)

        y_pred = gbm.predict(data_val, num_iteration=gbm.best_iteration)
        val_loss = log_loss(label_val, y_pred)
        val_loss_list.append(val_loss)

        joblib.dump(gbm, 'model/gbm_kfold_{}'.format(i))

        print 'train_loss: {:4}, val_loss: {:4}'.format(train_loss, val_loss)

    print train_loss_list
    print np.mean(train_loss_list)
    print val_loss_list
    print np.mean(val_loss_list)

    gbm1 = joblib.load('model/gbm_kfold_1')
    gbm2 = joblib.load('model/gbm_kfold_2')
    gbm3 = joblib.load('model/gbm_kfold_3')
    gbm4 = joblib.load('model/gbm_kfold_4')
    gbm5 = joblib.load('model/gbm_kfold_5')

    y_pred1 = gbm1.predict(data_val, num_iteration=gbm1.best_iteration)
    y_pred2 = gbm2.predict(data_val, num_iteration=gbm2.best_iteration)
    y_pred3 = gbm3.predict(data_val, num_iteration=gbm3.best_iteration)
    y_pred4 = gbm4.predict(data_val, num_iteration=gbm4.best_iteration)
    y_pred5 = gbm5.predict(data_val, num_iteration=gbm5.best_iteration)

    y = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5) / 5
    print 'val logloss: ', log_loss(label_val, y)

    
def submit():
    gbm1 = joblib.load('model/gbm_kfold_1')
    gbm2 = joblib.load('model/gbm_kfold_2')
    gbm3 = joblib.load('model/gbm_kfold_3')
    gbm4 = joblib.load('model/gbm_kfold_4')
    gbm5 = joblib.load('model/gbm_kfold_5')

    y_pred1 = gbm1.predict(data_test, num_iteration=gbm1.best_iteration)
    y_pred2 = gbm2.predict(data_test, num_iteration=gbm2.best_iteration)
    y_pred3 = gbm3.predict(data_test, num_iteration=gbm3.best_iteration)
    y_pred4 = gbm4.predict(data_test, num_iteration=gbm4.best_iteration)
    y_pred5 = gbm5.predict(data_test, num_iteration=gbm5.best_iteration)

    y = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5) / 5

    y = pd.Series(y, name='predicted_score')
    result = pd.concat([id_test, y], axis=1)
    time_format = '%Y-%m-%d-%H-%M-%S'
    time_now = datetime.datetime.now()
    result.to_csv('result/result_%s.csv'%time_now.strftime(time_format), index=False) # for backup
    result.to_csv('result/result.csv', index=False)
    print y.mean()


def main():
    # cv()
    submit()

if __name__ == '__main__':
    main()