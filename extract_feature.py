# coding=utf-8
import numpy as np
import pandas as pd
from sklearn import preprocessing
# import matplotlib.pyplot as plt

import datetime

# train_data : 2018-09-18 ~ 2018-09-24
# test_data : 2018-09-25

# sliding 2 days:
# train_data : 
#   dataset1: 2018-09-20 , features from 2018-09-18~2018-09-19
#   dataset2: 2018-09-21 , features from 2018-09-19~2018-09-20
#   dataset3: 2018-09-22 , features from 2018-09-20~2018-09-21
#   dataset4: 2018-09-23 , features from 2018-09-21~2018-09-22
# val_data :
#   dataset5: 2018-09-24 , features from 2018-09-22~2018-09-23
# test_data : 
#   dataset6: 2018-09-25 , features from 2018-09-23~2018-09-24


def main():
    train_table = pd.read_csv(
        'data/round1_ijcai_18_train_20180301.txt', sep=' ')
    train_table.replace(-1, np.NaN, inplace=True)
    test_table = pd.read_csv(
        'data/round1_ijcai_18_test_a_20180301.txt', sep=' ')
    test_table.replace(-1, np.NaN, inplace=True)

    # get time
    train_table['context_time'] = train_table['context_timestamp'].apply(
        lambda x: datetime.datetime.fromtimestamp(x).hour)
    test_table['context_time'] = test_table['context_timestamp'].apply(
        lambda x: datetime.datetime.fromtimestamp(x).hour)

    # get day
    train_table['context_day'] = train_table['context_timestamp'].apply(
        lambda x: datetime.datetime.fromtimestamp(x).day)
    test_table['context_day'] = test_table['context_timestamp'].apply(
        lambda x: datetime.datetime.fromtimestamp(x).day)

    instance_id = ['instance_id']
    item_feat_list = [
        'item_price_level', 'item_sales_level', 'item_collected_level',
        'item_pv_level'
    ]
    user_feat_list = [
        'user_gender_id', 'user_age_level', 'user_occupation_id',
        'user_star_level'
    ]
    cont_feat_list = ['context_time', 'context_page_id']
    shop_feat_list = [
        'shop_review_num_level', 'shop_review_positive_rate',
        'shop_star_level', 'shop_score_service', 'shop_score_delivery',
        'shop_score_description'
    ]
    label = ['is_trade']

    total_feat = instance_id + item_feat_list + user_feat_list + cont_feat_list + shop_feat_list

    train_table['user_age_level'] = train_table['user_age_level'] - 1000
    train_table[
        'user_occupation_id'] = train_table['user_occupation_id'] - 2000
    train_table['user_star_level'] = train_table['user_star_level'] - 3000
    train_table['context_page_id'] = train_table['context_page_id'] - 4000
    train_table['shop_star_level'] = train_table['shop_star_level'] - 5000

    test_table['user_age_level'] = test_table['user_age_level'] - 1000
    test_table['user_occupation_id'] = test_table['user_occupation_id'] - 2000
    test_table['user_star_level'] = test_table['user_star_level'] - 3000
    test_table['context_page_id'] = test_table['context_page_id'] - 4000
    test_table['shop_star_level'] = test_table['shop_star_level'] - 5000
    
    # df_list = []
    # for day in xrange(19, 24):
    #     df = train_table[train_table['context_day'] == day]
    #     df_feat = train_table[train_table['context_day'] == day-1 || train_table['context_day'] == day-2]
    #     # item_feature

    #     # user_feature

    #     # context_feature


    train_feat = train_table[total_feat + label]
    val_feat = val_table[total_feat + label]
    test_feat = test_table[total_feat]

    train_feat.to_csv('data/train_feat.csv', index=False)
    val_feat.to_csv('data/val_feat.csv', index=False)
    test_feat.to_csv('data/test_feat.csv', index=False)


if __name__ == '__main__':
    main()
