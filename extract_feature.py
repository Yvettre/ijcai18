# coding=utf-8
import numpy as np
import pandas as pd
from sklearn import preprocessing

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

def get_match_level(s):
    item_category_list = s['item_category_list'].split(';')
    item_property_list = s['item_property_list'].split(';')
    predict_category_property = s['predict_category_property'].split(';')
    max_match_level = 0
    for item in predict_category_property:
        match_level = 0
        cate = item.split(':')[0]
        if cate in item_category_list:
            match_level += (10 * (1+item_category_list.index(cate)))
            prop_list = item.split(':')[1].split(',')
            match_prop_set = set(prop_list).union(set(item_property_list))
            match_level += (10 * (len(match_prop_set) / float(len(prop_list))))
        if match_level > max_match_level:
            max_match_level = match_level
    return max_match_level


# 实验证明好像还是一小时的结果好一些
def get_time(s):
    h = datetime.datetime.fromtimestamp(s).hour
    return h
    # m = datetime.datetime.fromtimestamp(s).minute
    # if m >= 30:
    #     return h + 0.5
    # else:
    #     return h

def main():
    train_table = pd.read_csv(
        'data/round1_ijcai_18_train_20180301.txt', sep=' ')
    train_table.replace(-1, np.NaN, inplace=True)
    test_table = pd.read_csv(
        'data/round1_ijcai_18_test_a_20180301.txt', sep=' ')
    test_table.replace(-1, np.NaN, inplace=True)

    # get time
    train_table['context_time'] = train_table['context_timestamp'].apply(get_time)
    test_table['context_time'] = test_table['context_timestamp'].apply(get_time)

    # get day
    train_table['context_day'] = train_table['context_timestamp'].apply(
        lambda x: datetime.datetime.fromtimestamp(x).day)
    test_table['context_day'] = test_table['context_timestamp'].apply(
        lambda x: datetime.datetime.fromtimestamp(x).day)


    # basic feat
    instance_id = ['instance_id']
    item_feat_set = {
        'item_price_level', 'item_sales_level', 'item_collected_level',
        'item_pv_level'
    }
    user_feat_set = {
        'user_gender_id', 'user_age_level', 'user_occupation_id',
        'user_star_level'
    }
    cont_feat_set = {'context_time', 'context_page_id'}
    shop_feat_set = {
        'shop_review_num_level', 'shop_review_positive_rate',
        'shop_star_level', 'shop_score_service', 'shop_score_delivery',
        'shop_score_description'
    }
    cross_feat_set = set()
    leak_feat_set = set()
    label = ['is_trade']

    # # get match level
    train_table['item_match_level'] = train_table.apply(get_match_level, axis=1)
    test_table['item_match_level'] = test_table.apply(get_match_level, axis=1)
    item_feat_set.add('item_match_level')

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
    
    df_list = []
    for day in xrange(20, 26):
        print 'day: {}'.format(day)
        if day < 25:
            df = train_table[train_table['context_day'] == day].copy()
        else:
            df = test_table.copy()
        df_feat = train_table[(train_table['context_day'] == day-1) | (train_table['context_day'] == day-2)]
        # ==========================================================================================
        ## item_feature
        # item_trade_num
        item_feat_set.add('item_trade_num')
        item_trade_num = df_feat[['item_id', 'is_trade']]
        item_trade_num = item_trade_num.groupby('item_id').agg('sum').reset_index()
        item_trade_num.rename(columns={'is_trade': 'item_trade_num'}, inplace=True)
        df = pd.merge(df, item_trade_num, on=['item_id'], how='left')
        del item_trade_num
        # item_not_trade_num
        item_feat_set.add('item_not_trade_num')
        item_not_trade_num = df_feat[['item_id', 'is_trade']].copy()
        item_not_trade_num['is_trade'] = 1 - item_not_trade_num['is_trade']
        item_not_trade_num = item_not_trade_num.groupby('item_id').agg('sum').reset_index()
        item_not_trade_num.rename(columns={'is_trade': 'item_not_trade_num'}, inplace=True)
        df = pd.merge(df, item_not_trade_num, on=['item_id'], how='left')
        del item_not_trade_num
        # item_view_num
        item_feat_set.add('item_view_num')
        df['item_view_num'] = df['item_trade_num'] + df['item_not_trade_num']
        # item_trade_rate
        item_feat_set.add('item_trade_rate')
        df['item_trade_rate'] = df['item_trade_num'] / df['item_view_num']
        df['item_trade_rate'].fillna(0, inplace=True)

        # # item_brand_trade_num
        # item_feat_set.add('item_brand_trade_num')
        item_brand_trade_num = df_feat[['item_brand_id', 'is_trade']]
        item_brand_trade_num = item_brand_trade_num.groupby('item_brand_id').agg('sum').reset_index()
        item_brand_trade_num.rename(columns={'is_trade': 'item_brand_trade_num'}, inplace=True)
        df = pd.merge(df, item_brand_trade_num, on=['item_brand_id'], how='left')
        del item_brand_trade_num
        # # item_brand_not_trade_num
        item_feat_set.add('item_brand_not_trade_num')
        item_brand_not_trade_num = df_feat[['item_brand_id', 'is_trade']].copy()
        item_brand_not_trade_num['is_trade'] = 1 - item_brand_not_trade_num['is_trade']
        item_brand_not_trade_num = item_brand_not_trade_num.groupby('item_brand_id').agg('sum').reset_index()
        item_brand_not_trade_num.rename(columns={'is_trade': 'item_brand_not_trade_num'}, inplace=True)
        df = pd.merge(df, item_brand_not_trade_num, on=['item_brand_id'], how='left')
        del item_brand_not_trade_num
        # item_brand_view_num
        item_feat_set.add('item_brand_view_num')
        df['item_brand_view_num'] = df['item_brand_trade_num'] + df['item_brand_not_trade_num']
        # item_brand_trade_rate
        item_feat_set.add('item_brand_trade_rate')
        df['item_brand_trade_rate'] = df['item_brand_trade_num'] / df['item_brand_view_num']
        df['item_brand_trade_rate'].fillna(0, inplace=True)
        # ==========================================================================================
        ## shop_feature
        # shop_trade_num
        shop_feat_set.add('shop_trade_num')
        shop_trade_num = df_feat[['shop_id', 'is_trade']]
        shop_trade_num = shop_trade_num.groupby('shop_id').agg('sum').reset_index()
        shop_trade_num.rename(columns={'is_trade': 'shop_trade_num'}, inplace=True)        
        df = pd.merge(df, shop_trade_num, on=['shop_id'], how='left')
        del shop_trade_num
        # shop_not_trade_num
        shop_feat_set.add('shop_not_trade_num')
        shop_not_trade_num = df_feat[['shop_id', 'is_trade']].copy()
        shop_not_trade_num['is_trade'] = 1 - shop_not_trade_num['is_trade']
        shop_not_trade_num = shop_not_trade_num.groupby('shop_id').agg('sum').reset_index()
        shop_not_trade_num.rename(columns={'is_trade': 'shop_not_trade_num'}, inplace=True)
        df = pd.merge(df, shop_not_trade_num, on=['shop_id'], how='left')
        del shop_not_trade_num
        # shop_view_num
        shop_feat_set.add('shop_view_num')
        df['shop_view_num'] = df['shop_trade_num'] + df['shop_not_trade_num']
        # shop_trade_rate
        shop_feat_set.add('shop_trade_rate')
        df['shop_trade_rate'] = df['shop_trade_num'] / df['shop_view_num']
        df['shop_trade_rate'].fillna(0, inplace=True)
        # shop_score_service_inc
        shop_feat_set.add('shop_score_service_inc')        
        shop_score_service_past = df_feat[['shop_id', 'shop_score_service']]
        shop_score_service_past = shop_score_service_past.groupby('shop_id').agg('mean').reset_index()
        shop_score_service_past.rename(columns={'shop_score_service':'shop_score_service_past'}, inplace=True)
        df = pd.merge(df, shop_score_service_past, on=['shop_id'], how='left')
        df['shop_score_service_inc'] = 100.0 * (df['shop_score_service'] - df['shop_score_service_past']) / df['shop_score_service_past']
        df['shop_score_service_inc'].fillna(0, inplace=True)
        del shop_score_service_past
        # shop_score_delivery_inc
        shop_feat_set.add('shop_score_delivery_inc')        
        shop_score_delivery_past = df_feat[['shop_id', 'shop_score_delivery']]
        shop_score_delivery_past = shop_score_delivery_past.groupby('shop_id').agg('mean').reset_index()
        shop_score_delivery_past.rename(columns={'shop_score_delivery':'shop_score_delivery_past'}, inplace=True)
        df = pd.merge(df, shop_score_delivery_past, on=['shop_id'], how='left')
        df['shop_score_delivery_inc'] = 100.0 * (df['shop_score_delivery'] - df['shop_score_delivery_past']) / df['shop_score_delivery_past']
        df['shop_score_delivery_inc'].fillna(0, inplace=True)
        del shop_score_delivery_past
        # shop_score_description_inc
        shop_feat_set.add('shop_score_description_inc')        
        shop_score_description_past = df_feat[['shop_id', 'shop_score_description']]
        shop_score_description_past = shop_score_description_past.groupby('shop_id').agg('mean').reset_index()
        shop_score_description_past.rename(columns={'shop_score_description':'shop_score_description_past'}, inplace=True)
        df = pd.merge(df, shop_score_description_past, on=['shop_id'], how='left')
        df['shop_score_description_inc'] = 100.0 * (df['shop_score_description'] - df['shop_score_description_past']) / df['shop_score_description_past']
        df['shop_score_description_inc'].fillna(0, inplace=True)
        del shop_score_description_past
        # ==========================================================================================
        ## user_feature
        # user_gender_trade_num
        user_feat_set.add('user_gender_trade_num')
        user_gender_trade_num = df_feat[['user_gender_id', 'is_trade']]
        user_gender_trade_num = user_gender_trade_num.groupby('user_gender_id').agg('sum').reset_index()
        user_gender_trade_num.rename(columns={'is_trade': 'user_gender_trade_num'}, inplace=True)        
        df = pd.merge(df, user_gender_trade_num, on=['user_gender_id'], how='left')
        del user_gender_trade_num
        # user_gender_not_trade_num
        user_feat_set.add('user_gender_not_trade_num')
        user_gender_not_trade_num = df_feat[['user_gender_id', 'is_trade']].copy()
        user_gender_not_trade_num['is_trade'] = 1 - user_gender_not_trade_num['is_trade']        
        user_gender_not_trade_num = user_gender_not_trade_num.groupby('user_gender_id').agg('sum').reset_index()
        user_gender_not_trade_num.rename(columns={'is_trade': 'user_gender_not_trade_num'}, inplace=True)        
        df = pd.merge(df, user_gender_not_trade_num, on=['user_gender_id'], how='left')
        del user_gender_not_trade_num
        # user_gender_view_num
        user_feat_set.add('user_gender_view_num')
        df['user_gender_view_num'] = df['user_gender_trade_num'] + df['user_gender_not_trade_num']
        # user_gender_trade_rate
        user_feat_set.add('user_gender_trade_rate')
        df['user_gender_trade_rate'] = df['user_gender_trade_num'] / df['user_gender_view_num']
        df['user_gender_trade_rate'].fillna(0, inplace=True)
        # user_occupation_trade_num
        user_feat_set.add('user_occupation_trade_num')
        user_occupation_trade_num = df_feat[['user_occupation_id', 'is_trade']]
        user_occupation_trade_num = user_occupation_trade_num.groupby('user_occupation_id').agg('sum').reset_index()
        user_occupation_trade_num.rename(columns={'is_trade': 'user_occupation_trade_num'}, inplace=True)        
        df = pd.merge(df, user_occupation_trade_num, on=['user_occupation_id'], how='left')
        del user_occupation_trade_num
        # user_occupation_not_trade_num
        user_feat_set.add('user_occupation_not_trade_num')
        user_occupation_not_trade_num = df_feat[['user_occupation_id', 'is_trade']].copy()
        user_occupation_not_trade_num['is_trade'] = 1 - user_occupation_not_trade_num['is_trade']        
        user_occupation_not_trade_num = user_occupation_not_trade_num.groupby('user_occupation_id').agg('sum').reset_index()
        user_occupation_not_trade_num.rename(columns={'is_trade': 'user_occupation_not_trade_num'}, inplace=True)        
        df = pd.merge(df, user_occupation_not_trade_num, on=['user_occupation_id'], how='left')
        del user_occupation_not_trade_num
        # user_occupation_view_num
        user_feat_set.add('user_occupation_view_num')
        df['user_occupation_view_num'] = df['user_occupation_trade_num'] + df['user_occupation_not_trade_num']
        # user_occupation_trade_rate
        user_feat_set.add('user_occupation_trade_rate')
        df['user_occupation_trade_rate'] = df['user_occupation_trade_num'] / df['user_occupation_view_num']
        df['user_occupation_trade_rate'].fillna(0, inplace=True)
        # ==========================================================================================
        ## context_feature

        # ==========================================================================================
        # ## cross_featre
        # brand_gender_trade_num
        cross_feat_set.add('brand_gender_trade_num')
        brand_gender_trade_num = df_feat[['item_brand_id', 'user_gender_id', 'is_trade']]
        brand_gender_trade_num = brand_gender_trade_num.groupby(['item_brand_id', 'user_gender_id']).agg('sum').reset_index()
        brand_gender_trade_num.rename(columns={'is_trade': 'brand_gender_trade_num'}, inplace=True)
        df = pd.merge(df, brand_gender_trade_num, on=['item_brand_id', 'user_gender_id'], how='left')
        del brand_gender_trade_num
        # brand_gender_rate
        cross_feat_set.add('brand_gender_rate')
        df['brand_gender_rate'] = df['brand_gender_trade_num'] / df['item_brand_trade_num']
        df['brand_gender_rate'].fillna(0, inplace=True)
        # brand_age_trade_num
        cross_feat_set.add('brand_age_trade_num')
        brand_age_trade_num = df_feat[['item_brand_id', 'user_age_level', 'is_trade']]
        brand_age_trade_num = brand_age_trade_num.groupby(['item_brand_id', 'user_age_level']).agg('sum').reset_index()
        brand_age_trade_num.rename(columns={'is_trade': 'brand_age_trade_num'}, inplace=True)
        df = pd.merge(df, brand_age_trade_num, on=['item_brand_id', 'user_age_level'], how='left')
        del brand_age_trade_num
        # brand_age_rate
        cross_feat_set.add('brand_age_rate')
        df['brand_age_rate'] = df['brand_age_trade_num'] / df['item_brand_trade_num']
        df['brand_age_rate'].fillna(0, inplace=True) 
        # ==========================================================================================

        df_list.append(df)

    train_table = pd.concat(df_list[:-2], axis=0)
    val_table = df_list[-2]
    test_table = df_list[-1]
    
    total_feat = instance_id + list(item_feat_set) + list(user_feat_set) + list(cont_feat_set) + list(shop_feat_set) + list(cross_feat_set) + list(leak_feat_set)

    train_feat = train_table[total_feat + label]
    val_feat = val_table[total_feat + label]
    test_feat = test_table[total_feat]

    print train_table['context_day'][:10]
    print val_table['context_day'][:10]
    print test_table['context_day'][:10]

    train_feat.to_csv('data/train_feat.csv', index=False)
    val_feat.to_csv('data/val_feat.csv', index=False)
    test_feat.to_csv('data/test_feat.csv', index=False)


if __name__ == '__main__':
    main()
