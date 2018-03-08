#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('data/train_feat.csv')
val_df = pd.read_csv('data/val_feat.csv')
test_df = pd.read_csv('data/test_feat.csv')
train_df.fillna(0,inplace=True)
val_df.fillna(0,inplace=True)
label_train = train_df['is_trade'].values
data_train = train_df.drop(['instance_id', 'is_trade'], axis=1).values
label_val = val_df['is_trade'].values
data_val = val_df.drop(['instance_id', 'is_trade'], axis=1).values
id_test = test_df['instance_id']
data_test = test_df.drop(['instance_id'], axis=1).values

def log_loss(y_list, p_list):
    e = 1e-15
    assert len(y_list) == len(p_list), 'length not match {} vs. {}'.format(
        len(y_list), len(p_list))
    p_list = [x if x > e else e for x in p_list]
    p_list = [x if x < 1-e else 1-e for x in p_list]
    n = len(y_list)
    ans = 0
    for i in xrange(n):
        yi = y_list[i]
        pi = p_list[i]
        ans += yi * np.log(pi) + (1 - yi) * np.log(1 - pi)
    ans = -ans / n
    return ans

def main():
    model = LogisticRegression()
    model.fit(data_train, label_train)
    y_val = model.predict(data_val)
    y_train = model.predict(data_train)
    print log_loss(label_train, y_train)
    print log_loss(label_val, y_val)


if __name__ == '__main__':
    main() 