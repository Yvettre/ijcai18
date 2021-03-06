# coding=utf-8
# pylint:disable=E1101

import numpy as np
import pandas as pd
from sklearn import preprocessing

import datetime
import time

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

hour_offset = 12

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
            match_prop_set = set(prop_list) & set(item_property_list) # 取交集
            match_level += (10 * (len(match_prop_set) / float(len(prop_list))))
        if match_level > max_match_level:
            max_match_level = match_level

    return max_match_level

# 属性匹配比(考虑类别) = 商品属性列表与预测属性列表(考虑类别)匹配的属性个数 / 预测属性列表(考虑类别)属性个数
def get_prop_match_rate(s):
    item_category_list = s['item_category_list'].split(';')
    item_property_list = s['item_property_list'].split(';')
    predict_category_property = s['predict_category_property'].split(';')
    predict_prop_list = []

    try:
        for item in predict_category_property:
            cate = item.split(':')[0]
            if cate in item_category_list:
                prop_list = item.split(':')[1].split(',')
                predict_prop_list.extend(prop_list)
        match_prop_set = set(predict_prop_list) & set(item_property_list) # 取交集
        match_rate = len(match_prop_set) / float(len(predict_prop_list))
    except Exception,e:
        return np.NaN

    return match_rate

# 商品属性列表与预测属性列表(考虑类别)匹配的属性个数
def get_prop_match_num(s):
    item_category_list = s['item_category_list'].split(';')
    item_property_list = s['item_property_list'].split(';')
    predict_category_property = s['predict_category_property'].split(';')
    predict_prop_list = []

    try:
        for item in predict_category_property:
            cate = item.split(':')[0]
            if cate in item_category_list:
                prop_list = item.split(':')[1].split(',')
                predict_prop_list.extend(prop_list)
        match_prop_set = set(predict_prop_list) & set(item_property_list) # 取交集
    except Exception,e:
        return np.NaN
    return len(match_prop_set)
    
# 实验证明好像还是一小时的结果好一些
def get_time(s):
    h = datetime.datetime.fromtimestamp(s).hour
    return h
    # m = datetime.datetime.fromtimestamp(s).minute
    # if m >= 30:
    #     return 2*h + 1
    # else:
    #     return 2*h

def get_second_cate(x):
    cate_list = x.split(';')
    if len(cate_list) > 1:
        return cate_list[1]
    else:
        return '-1'

def get_caterank_in_predict(s):
    item_second_cate = s['item_second_cate']
    predict_category_property = s['predict_category_property'].split(';')

    flag = np.NaN
    for i,item in enumerate(predict_category_property):
        cate = item.split(':')[0]
        if cate==item_second_cate:
            flag = i
    return flag
    

def main():
    time_node1 = time.clock()

    train_table = pd.read_csv(
        'data/round2_train.txt', sep=' ')
    train_table.replace(-1, np.NaN, inplace=True)
    test_table = pd.read_csv(
        'data/round2_ijcai_18_test_a_20180425.txt', sep=' ')
    test_table.replace(-1, np.NaN, inplace=True)

    train_table.drop_duplicates(subset='instance_id', keep='first', inplace=True)
    test_table.drop_duplicates(subset='instance_id', keep='first', inplace=True)

    # get time
    train_table['context_time'] = train_table['context_timestamp'].apply(get_time)
    test_table['context_time'] = test_table['context_timestamp'].apply(get_time)

    # get day
    train_table['context_day'] = train_table['context_timestamp'].apply(
        lambda x: datetime.datetime.fromtimestamp(x).day)
    test_table['context_day'] = test_table['context_timestamp'].apply(
        lambda x: datetime.datetime.fromtimestamp(x).day)

    # get predict major cate
    train_table['predict_major_cate'] = train_table['predict_category_property'].apply(
        lambda x: x.split(':')[0])
    test_table['predict_major_cate'] = test_table['predict_category_property'].apply(
        lambda x: x.split(':')[0])
    le =preprocessing.LabelEncoder()
    le.fit(list(train_table['predict_major_cate']) + list(test_table['predict_major_cate']))
    train_table['predict_major_cate'] = le.transform(train_table['predict_major_cate'])
    test_table['predict_major_cate'] = le.transform(test_table['predict_major_cate'])
    del le 
    
    # get item major cate
    train_table['item_major_cate'] = train_table['item_category_list'].apply(
        lambda x: x.split(':')[0])
    test_table['item_major_cate'] = test_table['item_category_list'].apply(
        lambda x: x.split(':')[0])
    le = preprocessing.LabelEncoder()
    le.fit(list(train_table['item_major_cate']) + list(test_table['item_major_cate']))
    train_table['item_major_cate'] = le.transform(train_table['item_major_cate'])
    test_table['item_major_cate'] = le.transform(test_table['item_major_cate'])
    del le

    # get item second cate
    train_table['item_second_cate'] = train_table['item_category_list'].apply(get_second_cate)
    test_table['item_second_cate'] = test_table['item_category_list'].apply(get_second_cate)
    # get item_caterank_in_predict
    train_table['item_caterank_in_predict'] = train_table.apply(get_caterank_in_predict, axis=1)
    test_table['item_caterank_in_predict'] = test_table.apply(get_caterank_in_predict, axis=1)
    # LabelEncoder item_second_cate
    le = preprocessing.LabelEncoder()
    le.fit(list(train_table['item_second_cate']) + list(test_table['item_second_cate']))
    train_table['item_second_cate'] = le.transform(train_table['item_second_cate'])
    test_table['item_second_cate'] = le.transform(test_table['item_second_cate'])
    del le 

    # get item city id
    le = preprocessing.LabelEncoder()
    train_table['item_city_id'].replace(np.NaN, 0, inplace=True)
    test_table['item_city_id'].replace(np.NaN, 0, inplace=True)
    le.fit(list(train_table['item_city_id']) + list(test_table['item_city_id']))
    train_table['item_city_id'] = le.transform(train_table['item_city_id'])
    test_table['item_city_id'] = le.transform(test_table['item_city_id'])
    del le

    # get search keyword
    le = preprocessing.LabelEncoder()
    le.fit(list(train_table['predict_category_property']) + list(test_table['predict_category_property']))
    train_table['search_key'] = le.transform(train_table['predict_category_property'])
    test_table['search_key'] = le.transform(test_table['predict_category_property'])
    del le

    # get property_match_rate
    train_table['prop_match_rate'] = train_table.apply(get_prop_match_rate, axis=1)
    test_table['prop_match_rate'] = test_table.apply(get_prop_match_rate, axis=1)
    # get property_match_num
    train_table['prop_match_num'] = train_table.apply(get_prop_match_num, axis=1)
    test_table['prop_match_num'] = test_table.apply(get_prop_match_num, axis=1)

    # basic feat
    instance_id = ['instance_id']
    item_feat_set = {
        'item_price_level', 'item_sales_level', 'item_collected_level',
        'item_pv_level', 'item_city_id', 'item_major_cate','item_second_cate',
        'item_caterank_in_predict',
        'prop_match_rate','prop_match_num'
    }
    user_feat_set = {
        'user_gender_id', 'user_age_level', 'user_occupation_id',
        'user_star_level'
    }
    cont_feat_set = {'context_time', 'context_page_id', 'predict_major_cate'}# search_key 不要加，会过拟合
    shop_feat_set = {
        'shop_review_num_level', 'shop_review_positive_rate',
        'shop_star_level', 'shop_score_service', 'shop_score_delivery',
        'shop_score_description', 'shop_total_score'
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
    train_table['shop_total_score'] = 1.0*train_table['shop_score_service'] + 0.8*train_table['shop_score_delivery'] + 1.2*train_table['shop_score_description']

    test_table['user_age_level'] = test_table['user_age_level'] - 1000
    test_table['user_occupation_id'] = test_table['user_occupation_id'] - 2000
    test_table['user_star_level'] = test_table['user_star_level'] - 3000
    test_table['context_page_id'] = test_table['context_page_id'] - 4000
    test_table['shop_star_level'] = test_table['shop_star_level'] - 5000
    test_table['shop_total_score'] = 1.0*test_table['shop_score_service'] + 0.8*test_table['shop_score_delivery'] + 1.2*test_table['shop_score_description']

    time_node2 = time.clock()
    time_counter_tmp = time_node2 - time_node1
    print 'cost time before for loop: %dh-%dm-%ds'%(time_counter_tmp/3600, (time_counter_tmp%3600)/60, (time_counter_tmp%3600)%60)
    
    df_list = []
    # for day in xrange(20, 26):
    for item in xrange(2, 14):
        day = item/2 + 1
        noon_flag = item%2  # 0是早上，1是下午
        print 'day: {}, noon_flag: {}'.format(day, noon_flag)
        time_node3 = time.clock()
        
        # if day < 25:
        if day < 7 or (day == 7 and noon_flag == 0):
            # df = train_table[train_table['context_day'] == day].copy()
            if noon_flag==0:  # 早上
                df = train_table[(train_table['context_day'] == day)&(train_table['context_time']<hour_offset)].copy()
                # 复赛数据offset
                df_feat = train_table[(train_table['context_day'] == day-1) | (train_table['context_day']==day-2)]
            elif noon_flag==1:  #下午
                df = train_table[(train_table['context_day'] == day)&(train_table['context_time']>=hour_offset)].copy()
                # 复赛数据offset
                df_feat = train_table[((train_table['context_day'] == day)&(train_table['context_time']<hour_offset)) | (train_table['context_day'] == day-1) | ((train_table['context_day']==day-2)&(train_table['context_time']>=hour_offset))]
        elif day == 7 and noon_flag == 1:
            df = test_table.copy()
            # 复赛数据offset
            df_feat = train_table[(train_table['context_day'] == day) | (train_table['context_day'] == day-1) | ((train_table['context_day']==day-2)&(train_table['context_time']>=hour_offset))]
        # 初赛数据offset
        # df_feat = train_table[(train_table['context_day'] == day-1) | ((train_table['context_day']==day-2)&(train_table['context_time']>=hour_offset))]
        # df_feat = train_table[(train_table['context_day'] == day-1) | (train_table['context_day'] == day-2)]
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
        df['item_trade_rate'] = df['item_trade_num'] / (1 + df['item_view_num'])
        df['item_trade_rate'].fillna(0, inplace=True)
        # ------------------------------------------------------------------------------------------        
        # item_brand_trade_num
        item_feat_set.add('item_brand_trade_num')
        item_brand_trade_num = df_feat[['item_brand_id', 'is_trade']]
        item_brand_trade_num = item_brand_trade_num.groupby('item_brand_id').agg('sum').reset_index()
        item_brand_trade_num.rename(columns={'is_trade': 'item_brand_trade_num'}, inplace=True)
        df = pd.merge(df, item_brand_trade_num, on=['item_brand_id'], how='left')
        del item_brand_trade_num
        # item_brand_not_trade_num
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
        df['item_brand_trade_rate'] = df['item_brand_trade_num'] / (1 + df['item_brand_view_num'])
        df['item_brand_trade_rate'].fillna(0, inplace=True)
        # ------------------------------------------------------------------------------------------
        # item_cate_trade_num
        item_feat_set.add('item_cate_trade_num')
        item_cate_trade_num = df_feat[['item_second_cate', 'is_trade']]
        item_cate_trade_num = item_cate_trade_num.groupby('item_second_cate').agg('sum').reset_index()
        item_cate_trade_num.rename(columns={'is_trade': 'item_cate_trade_num'}, inplace=True)
        df = pd.merge(df, item_cate_trade_num, on=['item_second_cate'], how='left')
        del item_cate_trade_num
        # item_cate_not_trade_num
        item_feat_set.add('item_cate_not_trade_num')
        item_cate_not_trade_num = df_feat[['item_second_cate', 'is_trade']].copy()
        item_cate_not_trade_num['is_trade'] = 1 - item_cate_not_trade_num['is_trade']
        item_cate_not_trade_num = item_cate_not_trade_num.groupby('item_second_cate').agg('sum').reset_index()
        item_cate_not_trade_num.rename(columns={'is_trade': 'item_cate_not_trade_num'}, inplace=True)
        df = pd.merge(df, item_cate_not_trade_num, on=['item_second_cate'], how='left')
        del item_cate_not_trade_num
        # item_cate_view_num
        item_feat_set.add('item_cate_view_num')
        df['item_cate_view_num'] = df['item_cate_trade_num'] + df['item_cate_not_trade_num']
        # item_cate_trade_rate
        item_feat_set.add('item_cate_trade_rate')
        df['item_cate_trade_rate'] = df['item_cate_trade_num'] / (1 + df['item_cate_view_num'])
        df['item_cate_trade_rate'].fillna(0, inplace=True)
        # ------------------------------------------------------------------------------------------
        # item_rank_in_cate
        item_feat_set.add('item_rank_in_cate')
        tmp = df[['item_id', 'item_second_cate', 'item_trade_num']].copy()
        df['item_rank_in_cate'] = tmp['item_trade_num'].groupby([tmp['item_second_cate']]).rank(ascending=0, method='dense')
        del tmp
        # item_pct_in_cate
        item_feat_set.add('item_pct_in_cate')
        df['item_pct_in_cate'] = df['item_trade_num'] / (1 + df['item_cate_trade_num'])
        # item_rank_in_brand
        item_feat_set.add('item_rank_in_brand')
        tmp = df[['item_id', 'item_brand_id', 'item_trade_num']].copy()
        df['item_rank_in_brand'] = tmp['item_trade_num'].groupby([tmp['item_brand_id']]).rank(ascending=0, method='dense')
        del tmp
        # item_pct_in_brand
        item_feat_set.add('item_pct_in_brand')
        df['item_pct_in_brand'] = df['item_trade_num'] / (1 + df['item_brand_trade_num'])
        # ------------------------------------------------------------------------------------------
        # item_pricerank_in_cate
        item_feat_set.add('item_pricerank_in_cate')
        tmp = df[['item_id','item_second_cate','item_price_level']].copy()
        df['item_pricerank_in_cate'] = tmp['item_price_level'].groupby([tmp['item_second_cate']]).rank(ascending=0, method='dense')
        del tmp
        # brand_price_level_avg
        item_feat_set.add('brand_price_level_avg')
        tmp = df_feat[['item_brand_id', 'item_price_level']].groupby('item_brand_id')
        brand_price_level_avg = tmp.mean().reset_index()
        brand_price_level_avg.rename(columns={'item_price_level':'brand_price_level_avg'}, inplace=True)
        df = pd.merge(df, brand_price_level_avg, on=['item_brand_id'], how='left')
        del brand_price_level_avg
        # above_brand_price_level_avg_value
        item_feat_set.add('above_brand_price_level_avg_value')
        df['above_brand_price_level_avg_value'] = df['item_price_level'] - df['brand_price_level_avg']
        # above_brand_price_level_avg_rate
        item_feat_set.add('above_brand_price_level_avg_rate')
        df['above_brand_price_level_avg_rate'] = df['above_brand_price_level_avg_value'] / df['brand_price_level_avg']
        # brand_price_level_min
        item_feat_set.add('brand_price_level_min')
        brand_price_level_min = tmp.min().reset_index()
        brand_price_level_min.rename(columns={'item_price_level':'brand_price_level_min'}, inplace=True)
        df = pd.merge(df, brand_price_level_min, on=['item_brand_id'], how='left')
        del brand_price_level_min
        # brand_price_level_max
        item_feat_set.add('brand_price_level_max')
        brand_price_level_max = tmp.max().reset_index()
        brand_price_level_max.rename(columns={'item_price_level':'brand_price_level_max'}, inplace=True)
        df = pd.merge(df, brand_price_level_max, on=['item_brand_id'], how='left')
        del brand_price_level_max
        # brand_price_level_std
        item_feat_set.add('brand_price_level_std')
        brand_price_level_std = tmp.std().reset_index()
        brand_price_level_std.rename(columns={'item_price_level':'brand_price_level_std'}, inplace=True)
        df = pd.merge(df, brand_price_level_std, on=['item_brand_id'], how='left')
        del brand_price_level_std
        # brand_price_level_median
        item_feat_set.add('brand_price_level_median')
        brand_price_level_median = tmp.median().reset_index()
        brand_price_level_median.rename(columns={'item_price_level':'brand_price_level_median'}, inplace=True)
        df = pd.merge(df, brand_price_level_median, on=['item_brand_id'], how='left')
        del brand_price_level_median
        # brand_price_level_mode
        item_feat_set.add('brand_price_level_mode')
        brand_price_level_mode = tmp.agg(lambda x: np.mean(pd.Series.mode(x))).reset_index()
        brand_price_level_mode.rename(columns={'item_price_level':'brand_price_level_mode'}, inplace=True)
        df = pd.merge(df, brand_price_level_mode, on=['item_brand_id'], how='left')
        del brand_price_level_mode
        del tmp
        # ------------------------------------------------------------------------------------------
        # cate_price_level_avg
        item_feat_set.add('cate_price_level_avg')
        tmp = df_feat[['item_second_cate', 'item_price_level']].groupby('item_second_cate')
        cate_price_level_avg = tmp.mean().reset_index()
        cate_price_level_avg.rename(columns={'item_price_level':'cate_price_level_avg'}, inplace=True)
        df = pd.merge(df, cate_price_level_avg, on=['item_second_cate'], how='left')
        del cate_price_level_avg
        # above_cate_price_level_avg_value
        item_feat_set.add('above_cate_price_level_avg_value')
        df['above_cate_price_level_avg_value'] = df['item_price_level'] - df['cate_price_level_avg']
        # above_cate_price_level_avg_rate
        item_feat_set.add('above_cate_price_level_avg_rate')
        df['above_cate_price_level_avg_rate'] = df['above_cate_price_level_avg_value'] / df['cate_price_level_avg']
        # cate_price_level_min
        item_feat_set.add('cate_price_level_min')
        cate_price_level_min = tmp.min().reset_index()
        cate_price_level_min.rename(columns={'item_price_level':'cate_price_level_min'}, inplace=True)
        df = pd.merge(df, cate_price_level_min, on=['item_second_cate'], how='left')
        del cate_price_level_min
        # cate_price_level_max
        item_feat_set.add('cate_price_level_max')
        cate_price_level_max = tmp.max().reset_index()
        cate_price_level_max.rename(columns={'item_price_level':'cate_price_level_max'}, inplace=True)
        df = pd.merge(df, cate_price_level_max, on=['item_second_cate'], how='left')
        del cate_price_level_max
        # cate_price_level_std
        item_feat_set.add('cate_price_level_std')
        cate_price_level_std = tmp.std().reset_index()
        cate_price_level_std.rename(columns={'item_price_level':'cate_price_level_std'}, inplace=True)
        df = pd.merge(df, cate_price_level_std, on=['item_second_cate'], how='left')
        del cate_price_level_std
        # cate_price_level_median
        item_feat_set.add('cate_price_level_median')
        cate_price_level_median = tmp.median().reset_index()
        cate_price_level_median.rename(columns={'item_price_level':'cate_price_level_median'}, inplace=True)
        df = pd.merge(df, cate_price_level_median, on=['item_second_cate'], how='left')
        del cate_price_level_median
        # cate_price_level_mode
        item_feat_set.add('cate_price_level_mode')
        cate_price_level_mode = tmp.agg(lambda x: np.mean(pd.Series.mode(x))).reset_index()
        cate_price_level_mode.rename(columns={'item_price_level':'cate_price_level_mode'}, inplace=True)
        df = pd.merge(df, cate_price_level_mode, on=['item_second_cate'], how='left')
        del cate_price_level_mode
        del tmp
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
        df['shop_trade_rate'] = df['shop_trade_num'] / (1 + df['shop_view_num'])
        df['shop_trade_rate'].fillna(0, inplace=True)
        # ------------------------------------------------------------------------------------------
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
        # ------------------------------------------------------------------------------------------
        shop_feat_set.add('shop_price_level_avg')
        shop_price_level_avg = df[['shop_id','item_price_level']]
        shop_price_level_avg = shop_price_level_avg.groupby('shop_id').agg('mean').reset_index()
        shop_price_level_avg.rename(columns={'item_price_level':'shop_price_level_avg'}, inplace=True)
        df = pd.merge(df, shop_price_level_avg, on=['shop_id'], how='left')
        del shop_price_level_avg
        # above_shop_price_level_avg_value
        item_feat_set.add('above_shop_price_level_avg_value')
        df['above_shop_price_level_avg_value'] = df['item_price_level'] - df['shop_price_level_avg']
        # above_shop_price_level_avg_rate
        item_feat_set.add('above_shop_price_level_avg_rate')
        df['above_shop_price_level_avg_rate'] = df['above_shop_price_level_avg_value'] / df['shop_price_level_avg']
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
        df['user_gender_trade_rate'] = df['user_gender_trade_num'] / (1 + df['user_gender_view_num'])
        df['user_gender_trade_rate'].fillna(0, inplace=True)
        # ------------------------------------------------------------------------------------------
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
        df['user_occupation_trade_rate'] = df['user_occupation_trade_num'] / (1 + df['user_occupation_view_num'])
        df['user_occupation_trade_rate'].fillna(0, inplace=True)
        # ------------------------------------------------------------------------------------------   
        # user_age_trade_num
        user_feat_set.add('user_age_trade_num')
        user_age_trade_num = df_feat[['user_age_level', 'is_trade']]
        user_age_trade_num = user_age_trade_num.groupby('user_age_level').agg('sum').reset_index()
        user_age_trade_num.rename(columns={'is_trade': 'user_age_trade_num'}, inplace=True)        
        df = pd.merge(df, user_age_trade_num, on=['user_age_level'], how='left')
        del user_age_trade_num
        # user_age_not_trade_num
        user_feat_set.add('user_age_not_trade_num')
        user_age_not_trade_num = df_feat[['user_age_level', 'is_trade']].copy()
        user_age_not_trade_num['is_trade'] = 1 - user_age_not_trade_num['is_trade']        
        user_age_not_trade_num = user_age_not_trade_num.groupby('user_age_level').agg('sum').reset_index()
        user_age_not_trade_num.rename(columns={'is_trade': 'user_age_not_trade_num'}, inplace=True)        
        df = pd.merge(df, user_age_not_trade_num, on=['user_age_level'], how='left')
        del user_age_not_trade_num
        # user_age_view_num
        user_feat_set.add('user_age_view_num')
        df['user_age_view_num'] = df['user_age_trade_num'] + df['user_age_not_trade_num']
        # user_age_trade_rate
        user_feat_set.add('user_age_trade_rate')
        df['user_age_trade_rate'] = df['user_age_trade_num'] / (1 + df['user_age_view_num'])
        df['user_age_trade_rate'].fillna(0, inplace=True)
        # ------------------------------------------------------------------------------------------     
        # user_age_view_price_level_avg
        user_feat_set.add('user_age_view_price_level_avg')
        tmp = df_feat[['user_age_level', 'item_price_level']].groupby('user_age_level')
        user_age_view_price_level_avg = tmp.mean().reset_index()
        user_age_view_price_level_avg.rename(columns={'item_price_level':'user_age_view_price_level_avg'}, inplace=True)
        df = pd.merge(df, user_age_view_price_level_avg, on=['user_age_level'], how='left')
        del user_age_view_price_level_avg
        # above_user_view_price_level_avg_value
        user_feat_set.add('above_user_view_price_level_avg_value')
        df['above_user_view_price_level_avg_value'] = df['user_age_view_price_level_avg'] - df['user_age_view_price_level_avg']
        # above_user_view_price_level_avg_rate
        user_feat_set.add('above_user_view_price_level_avg_rate')
        df['above_user_view_price_level_avg_rate'] = df['above_user_view_price_level_avg_value'] / df['user_age_view_price_level_avg']
        # user_age_view_price_level_min
        user_feat_set.add('user_age_view_price_level_min')
        user_age_view_price_level_min = tmp.min().reset_index()
        user_age_view_price_level_min.rename(columns={'item_price_level':'user_age_view_price_level_min'}, inplace=True)
        df = pd.merge(df, user_age_view_price_level_min, on=['user_age_level'], how='left')
        del user_age_view_price_level_min
        # user_age_view_price_level_max
        user_feat_set.add('user_age_view_price_level_max')
        user_age_view_price_level_max = tmp.max().reset_index()
        user_age_view_price_level_max.rename(columns={'item_price_level':'user_age_view_price_level_max'}, inplace=True)
        df = pd.merge(df, user_age_view_price_level_max, on=['user_age_level'], how='left')
        del user_age_view_price_level_max
        # user_age_view_price_level_std
        user_feat_set.add('user_age_view_price_level_std')
        user_age_view_price_level_std = tmp.std().reset_index()
        user_age_view_price_level_std.rename(columns={'item_price_level':'user_age_view_price_level_std'}, inplace=True)
        df = pd.merge(df, user_age_view_price_level_std, on=['user_age_level'], how='left')
        del user_age_view_price_level_std
        # user_age_view_price_level_median
        user_feat_set.add('user_age_view_price_level_median')
        user_age_view_price_level_median = tmp.median().reset_index()
        user_age_view_price_level_median.rename(columns={'item_price_level':'user_age_view_price_level_median'}, inplace=True)
        df = pd.merge(df, user_age_view_price_level_median, on=['user_age_level'], how='left')
        del user_age_view_price_level_median
        # user_age_view_price_level_mode
        user_feat_set.add('user_age_view_price_level_mode')
        user_age_view_price_level_mode = tmp.agg(lambda x: np.mean(pd.Series.mode(x))).reset_index()
        user_age_view_price_level_mode.rename(columns={'item_price_level':'user_age_view_price_level_mode'}, inplace=True)
        df = pd.merge(df, user_age_view_price_level_mode, on=['user_age_level'], how='left')
        del user_age_view_price_level_mode
        del tmp
        # ==========================================================================================
        ## context_feature
        # time_trade_num
        cont_feat_set.add('time_trade_num')
        time_trade_num = df_feat[['context_time', 'is_trade']]
        time_trade_num = time_trade_num.groupby('context_time').agg('sum').reset_index()
        time_trade_num.rename(columns={'is_trade': 'time_trade_num'}, inplace=True)        
        df = pd.merge(df, time_trade_num, on=['context_time'], how='left')
        del time_trade_num
        # time_not_trade_num
        cont_feat_set.add('time_not_trade_num')
        time_not_trade_num = df_feat[['context_time', 'is_trade']].copy()
        time_not_trade_num['is_trade'] = 1 - time_not_trade_num['is_trade']
        time_not_trade_num = time_not_trade_num.groupby('context_time').agg('sum').reset_index()
        time_not_trade_num.rename(columns={'is_trade': 'time_not_trade_num'}, inplace=True)
        df = pd.merge(df, time_not_trade_num, on=['context_time'], how='left')
        del time_not_trade_num
        # time_view_num
        cont_feat_set.add('time_view_num')
        df['time_view_num'] = df['time_trade_num'] + df['time_not_trade_num']
        # time_trade_rate
        cont_feat_set.add('time_trade_rate')
        df['time_trade_rate'] = df['time_trade_num'] / (1 + df['time_view_num'])
        df['time_trade_rate'].fillna(0, inplace=True)
        # ------------------------------------------------------------------------------------------
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
        df['brand_gender_rate'] = df['brand_gender_trade_num'] / (1 + df['item_brand_trade_num'])
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
        df['brand_age_rate'] = df['brand_age_trade_num'] / (1 + df['item_brand_trade_num'])
        df['brand_age_rate'].fillna(0, inplace=True)
        # user_view_item_num
        cross_feat_set.add('user_view_item_num')
        tmp = df_feat[['user_id', 'item_id']]
        tmp = tmp.groupby(['user_id','item_id']).size().reset_index()
        tmp.rename(columns={0:'user_view_item_num'}, inplace=True)
        df = pd.merge(df, tmp, on=['user_id', 'item_id'], how='left')
        del tmp
        # user_view_cate_num
        cross_feat_set.add('user_view_cate_num')
        tmp = df_feat[['user_id', 'item_second_cate']]
        tmp = tmp.groupby(['user_id','item_second_cate']).size().reset_index()
        tmp.rename(columns={0:'user_view_cate_num'}, inplace=True)        
        df = pd.merge(df, tmp, on=['user_id', 'item_second_cate'], how='left')
        del tmp
        # user_view_brand_num
        cross_feat_set.add('user_view_brand_num')
        tmp = df_feat[['user_id', 'item_brand_id']]
        tmp = tmp.groupby(['user_id','item_brand_id']).size().reset_index()
        tmp.rename(columns={0:'user_view_brand_num'}, inplace=True)        
        df = pd.merge(df, tmp, on=['user_id', 'item_brand_id'], how='left')
        del tmp  
        # ==========================================================================================
        ## leakage feature
        # today_shop_view_num
        leak_feat_set.add('today_shop_view_num')
        today_shop_view_num = df[['shop_id']].copy()
        today_shop_view_num['today_shop_view_num'] = 1
        today_shop_view_num = today_shop_view_num.groupby('shop_id').agg('sum').reset_index()
        df = pd.merge(df, today_shop_view_num, on=['shop_id'], how='left')
        del today_shop_view_num
        # today_item_view_num
        leak_feat_set.add('today_item_view_num')
        today_item_view_num = df[['item_id']].copy()
        today_item_view_num['today_item_view_num'] = 1
        today_item_view_num = today_item_view_num.groupby('item_id').agg('sum').reset_index()
        df = pd.merge(df, today_item_view_num, on=['item_id'], how='left')
        del today_item_view_num
        # today_user_view_shop_rev_time
        # 当前访问是今天用户访问该商店的倒数第几次
        leak_feat_set.add('today_user_view_shop_rev_time')
        leak_feat_set.add('today_user_view_shop_time')
        leak_feat_set.add('today_user_view_shop_num')
        leak_feat_set.add('today_user_view_shop_pct')
        tmp = df[['user_id', 'shop_id', 'context_timestamp']].copy()
        # 分组排序，有点蛋疼，记住这个写法
        df['today_user_view_shop_rev_time'] = tmp['context_timestamp'].groupby([tmp['user_id'], tmp['shop_id']]).rank(ascending=0, method='dense')
        df['today_user_view_shop_time'] = tmp['context_timestamp'].groupby([tmp['user_id'], tmp['shop_id']]).rank(ascending=1, method='dense')
        df['today_user_view_shop_num'] = df['today_user_view_shop_time'] + df['today_user_view_shop_rev_time'] - 1
        df['today_user_view_shop_pct'] = df['today_user_view_shop_time'] / (1 + df['today_user_view_shop_num'])
        del tmp
        # today_user_view_item_rev_time
        # 当前访问是今天用户访问该商品的倒数第几次
        leak_feat_set.add('today_user_view_item_rev_time')
        leak_feat_set.add('today_user_view_item_time')
        leak_feat_set.add('today_user_view_item_num')
        leak_feat_set.add('today_user_view_item_pct')
        tmp = df[['user_id', 'item_id', 'context_timestamp']].copy()
        df['today_user_view_item_rev_time'] = tmp['context_timestamp'].groupby([tmp['user_id'], tmp['item_id']]).rank(ascending=0, method='dense')
        df['today_user_view_item_time'] = tmp['context_timestamp'].groupby([tmp['user_id'], tmp['item_id']]).rank(ascending=1, method='dense')
        df['today_user_view_item_num'] = df['today_user_view_item_time'] + df['today_user_view_item_rev_time'] - 1
        df['today_user_view_item_pct'] = df['today_user_view_item_time'] / (1 + df['today_user_view_item_num'])
        del tmp
        # today_user_view_brand_rev_time        
        # 当前访问是今天用户访问该品牌的倒数第几次
        leak_feat_set.add('today_user_view_brand_rev_time')
        leak_feat_set.add('today_user_view_brand_time')
        leak_feat_set.add('today_user_view_brand_num')
        leak_feat_set.add('today_user_view_brand_pct')
        tmp = df[['user_id', 'item_brand_id', 'context_timestamp']].copy()
        df['today_user_view_brand_rev_time'] = tmp['context_timestamp'].groupby([tmp['user_id'], tmp['item_brand_id']]).rank(ascending=0, method='dense')
        df['today_user_view_brand_time'] = tmp['context_timestamp'].groupby([tmp['user_id'], tmp['item_brand_id']]).rank(ascending=1, method='dense')
        df['today_user_view_brand_num'] = df['today_user_view_brand_time'] + df['today_user_view_brand_rev_time'] - 1
        df['today_user_view_brand_pct'] = df['today_user_view_brand_time'] / (1 + df['today_user_view_brand_num'])
        del tmp
        # today_user_view_cate_rev_time        
        # 当前访问是今天用户访问该类别商品的倒数第几次
        leak_feat_set.add('today_user_view_cate_rev_time')
        leak_feat_set.add('today_user_view_cate_time')
        leak_feat_set.add('today_user_view_cate_num')
        leak_feat_set.add('today_user_view_cate_pct')
        tmp = df[['user_id', 'predict_major_cate', 'context_timestamp']].copy()
        df['today_user_view_cate_rev_time'] = tmp['context_timestamp'].groupby([tmp['user_id'], tmp['predict_major_cate']]).rank(ascending=0, method='dense')
        df['today_user_view_cate_time'] = tmp['context_timestamp'].groupby([tmp['user_id'], tmp['predict_major_cate']]).rank(ascending=1, method='dense')
        df['today_user_view_cate_num'] = df['today_user_view_cate_time'] + df['today_user_view_cate_rev_time'] - 1
        df['today_user_view_cate_pct'] = df['today_user_view_cate_time'] / (1 + df['today_user_view_cate_num'])

        leak_feat_set.add('today_user_view_item_cate_rev_time')
        leak_feat_set.add('today_user_view_item_cate_time')
        leak_feat_set.add('today_user_view_item_cate_num')
        leak_feat_set.add('today_user_view_item_cate_pct')
        tmp = df[['user_id', 'item_major_cate', 'context_timestamp']].copy()
        df['today_user_view_item_cate_rev_time'] = tmp['context_timestamp'].groupby([tmp['user_id'], tmp['item_major_cate']]).rank(ascending=0, method='dense')
        df['today_user_view_item_cate_time'] = tmp['context_timestamp'].groupby([tmp['user_id'], tmp['item_major_cate']]).rank(ascending=1, method='dense')
        df['today_user_view_item_cate_num'] = df['today_user_view_item_cate_time'] + df['today_user_view_item_cate_rev_time'] - 1
        df['today_user_view_item_cate_pct'] = df['today_user_view_item_cate_time'] / (1 + df['today_user_view_item_cate_num'])
        del tmp
        # today_user_view_keyword_rev_time
        # 当前访问的关键词广告是用户今天访问的倒数第几次
        leak_feat_set.add('today_user_view_keyword_rev_time')
        leak_feat_set.add('today_user_view_keyword_time')
        leak_feat_set.add('today_user_view_keyword_num')
        leak_feat_set.add('today_user_view_keyword_pct')
        tmp = df[['user_id', 'search_key', 'context_timestamp']].copy()
        df['today_user_view_keyword_rev_time'] = tmp['context_timestamp'].groupby([tmp['user_id'], tmp['search_key']]).rank(ascending=0, method='dense')
        df['today_user_view_keyword_time'] = tmp['context_timestamp'].groupby([tmp['user_id'], tmp['search_key']]).rank(ascending=1, method='dense')
        df['today_user_view_keyword_num'] = df['today_user_view_keyword_time'] + df['today_user_view_keyword_rev_time'] - 1
        df['today_user_view_keyword_pct'] = df['today_user_view_keyword_time'] / (1 + df['today_user_view_keyword_num'])
        # ------------------------------------------------------------------------------------------
        leak_feat_set.add('today_user_query_hour')
        tmp = df[['user_id','context_time']].copy()
        tmp = tmp.groupby(['user_id', 'context_time']).size().reset_index().rename(columns={0:'today_user_query_hour'})
        df = pd.merge(df, tmp, on=['user_id', 'context_time'], how='left')
        del tmp
        leak_feat_set.add('today_item_query_hour')
        tmp = df[['item_id','context_time']].copy()
        tmp = tmp.groupby(['item_id','context_time']).size().reset_index().rename(columns={0:'today_item_query_hour'})
        df = pd.merge(df, tmp, on=['item_id','context_time'], how='left')
        del tmp
        leak_feat_set.add('today_shop_query_hour')
        tmp = df[['shop_id','context_time']].copy()
        tmp = tmp.groupby(['shop_id','context_time']).size().reset_index().rename(columns={0:'today_shop_query_hour'})
        df = pd.merge(df, tmp, on=['shop_id','context_time'], how='left')
        del tmp
        leak_feat_set.add('today_cate_query_hour')
        tmp = df[['item_second_cate','context_time']].copy()
        tmp = tmp.groupby(['item_second_cate','context_time']).size().reset_index().rename(columns={0:'today_cate_query_hour'})
        df = pd.merge(df, tmp, on=['item_second_cate','context_time'], how='left')
        del tmp
        # ==========================================================================================
        df_list.append(df)

        time_node4 = time.clock()
        time_counter_tmp = time_node4 - time_node3
        print '%d day %d noon cost time: %dh-%dm-%ds'%(day, noon_flag, time_counter_tmp/3600, (time_counter_tmp%3600)/60, (time_counter_tmp%3600)%60)

        

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

    train_feat.to_csv('data/qing/train_feat.csv', index=False)
    val_feat.to_csv('data/qing/val_feat.csv', index=False)
    test_feat.to_csv('data/qing/test_feat.csv', index=False)

    time_node5 = time.clock()
    time_counter_tmp = time_node5 - time_node1
    print 'extract_features.py cost time: %dh-%dm-%ds'%(time_counter_tmp/3600, (time_counter_tmp%3600)/60, (time_counter_tmp%3600)%60)


if __name__ == '__main__':
    main()
