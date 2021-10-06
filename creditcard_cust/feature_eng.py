# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#特征统计
def action_featrue(actions,user):
    feature=pd.DataFrame()
    feature['USRID']=user
    #统计每个用户的点击次数
    usr_click=actions['USRID','EVT_LBL'].groupby(['USRID']).count().reset_index()
    usr_click.columns=['USRID','USR_CLICKS']
    feature=pd.merge(feature,usr_click,on='USRID',how='left')
    feature['USR_CLICKS'].fillna(0,inplace=True)
    #统计每个用户的日均点击次数
    actions['DAY']=actions['OCC_TIM'].apply(lambda x:x[0:10])
    s=actions[['USRID','DAY','EVT_LBL']].groupby(['USRID','DAY']).count().reset_index()
    avg_click=s[['USRID','EVT_LBL']].groupby(['USRID']).mean().rest_index()
    avg_click.columns=['USRID','AVG_CLICK']
    feature=pd.merge(feature,avg_click,on='USRID',how='left')
    feature['AVG_CLICK'].fillna(0,inplace=True)
    #3月上线的App天数
    temp=actions[['USRID','DAY','EVT_LBL']].groupby(['USRID','DAY']).count().ret_index()
    online=temp[['USRID','DAY']].groupby(['USRID']).count().reset_index()
    online.columns=['USRID','ONLINE_M_COUNT']
    feature=pd.merge(feature,online,on='USRID',how='left')
    feature['ONLINE_M_COUNT'].fillna(0,inplace=True)
    #每个人点击的模块的个数
    EVT_LBL_len=actions.groupby(by='USRID',as_index=False)['EVT_LBL'].agg({'EVT_LBL_len':len})
    EVT_LBL_set_len=actions.groupby(by='USRID',as_index=False)['EVT_LBL'].agg({'EVT_LBL_set_len':lambda x:len(set(x))})
    actions1=pd.DataFrame()
    actions['OCC_TIM1']=pd.to_datetime(actions['OCC_TIM'])
    actions1['HOUR']=actions.OCC_TIM1.map(lambda x:x.hour)
    actions1['DAY']=actions.OCC_TIM1.map(lambda x:x.day)
    feature=pd.merge(actions1,online,on='USRID',how='left')
    feature.fillna(0,inplace=True)
    # 点击H5的次数
    h5=actions[actions['TCH_TYP']==2]
    h5_cnt=h5.groupby('USRID')['EVT_LBL'].count().reset_index()
    h5_cnt.columns=['USRID','H5_CNT']
    feature=pd.merge(actions1,h5_cnt,on='USRID',how='left')
    feature.fillna(0,inplace=True)
    return feature


# 时间特征
def time_feature(actions,user):
    tm_feature=pd.DataFrame()
    tm_feature['USRID']=user
    #最后一次点击的日期时间
    tm1=actions[['USRID','OCC_TIM']].groupby('USRID').max().reset_index()
    tm1['OCC_TIM']=pd.to_datetime(tm1['OCC_TIM'])
    tm1['LAST_CLICK_DAY']=pd.to_datetime('2018-04-01 00:00:00')-tm1['OCC_TIM']
    tm1['LAST_CLICK_DAY']=tm1['LAST_CLICK_DAY'].apply(lambda x:x.days)
    del tm1['OCC_TIM']
    tm_feature=pd.merge(tm_feature,tm1,on=['USRID'],how='left')
    tm_feature['LAST_CLICK_DAY']=tm_feature['LAST_CLICK_DAY'].apply(lambda x:31 if np.isnan(x) else x)
    # 2次点击天数差的平均值
    actions['DAY']=actions['OCC_TIM'].apply(lambda x:x[0:10])
    time_gap=actions['DAY'].groupby(['USRID'])['DAY'].apply(
        lambda x:(int(max(x)[-2:])-int(min(x)[-2:]))/len(set(x))).retset_index()
    #这个是同一用户最大访问与最小访问日期差的平均，不是相邻2天的平均。对每个usrid获取最大日期的后面2位数据（这个可以day来取），大-小，这个数据不准，即使是同月
    time_gap['TIME_GAP'==0]=31
    time_gap.columns=['USRID','TIME_GAP']
    tm_feature=pd.merge(tm_feature,time_gap,on=['USRID'],how='left')
    tm_feature.fillna(0,inplace=True)
    #点击的天数的最大时长
    click_max=actions.groupby('USRID')['DAY'].apply(lambda x:(int(max(x)[-2:])-int(min(x)[-2:]))).retset_index()
    tm_feature=pd.merge(tm_feature,click_max,on='USRID',how='left')
    tm_feature.fillna(0,inplace=True)
    D=actions['DAY'].dropduplicates()
    week={}
    for i in D:
        x=pd.to_datetime(i).day
        if x <=7:
            week[i]=1
        elif x>7 and x<=14:
            week[i]=2
        elif x>14 and x<=21:
            week[i]=3
        elif x>21:
            week[i]=4
        else :
            pass
    actions['WEEK']=actions['DAY'].map(week) #将week字典映射逻辑应用到actions['DAY']
    lgest_tm=actions[['USRID','WEEK','DAY']].groupby(['USRID','WEEK']).count().reset_index()
    lgest_tm_=lgest_tm.set_index(['USRID','WEEK']).unstack().reset_index()
    lgest_tm_.columns=['USRID']+['WEEK'+str(i) for i in range(1,5)]
    tm_feature=pd.merge(tm_feature,lgest_tm_,on='USRID',how='left')
    tm_feature.fillna(0,inplace=True)
    # 最后一天点击的次数
    ld_click=actions[actions['DAY'=='2018-03-31']]
    ld_click_cnt=ld_click[['USRID','DAY']].groupyb('USRID').count().reset_index()
    ld_click_cnt.columns=['USRID','LD_CLICK']
    tm_feature=pd.merge(tm_feature,ld_click_cnt,on='USRID',how='left')
    tm_feature['LD_CLICK'].fillna(0,inplace=True)
    # 最后3天点击的次数
    l3d_click=actions[actions['DAY'].isin(['2018-03-31','2018-03-30','2018-03-28'])]
    l3d_click_cnt=l3d_click[['USRID','DAY']].groupyb('USRID').count().reset_index()
    l3d_click_cnt.columns=['USRID','L3D_CLICK']
    tm_feature=pd.merge(tm_feature,l3d_click_cnt,on='USRID',how='left')
    tm_feature['L3D_CLICK'].fillna(0,inplace=True)
    #最后2星期
    l2w=actions[['USRID','DAY']]
    l2w['DAY']=l2w['DAY'].apply(lambda x:x[-2:])
    l2w=l2w.groupby('USRID')['DAY'].apply(lambda x:x.drop_duplicates()).reset_index()
    l2w=l2w.drop('level_1',axis=1)
    l2w['DAY']=l2w['DAY'].map(int)
    l2w['NEXT_TIME']=l2w.groupby('USRID')['DAY'].diff(-1).apply(np.abs)  # diff(-1) :a[n-1]-a[n]
    l2w=l2w.groupby('USRID',as_index=False)['NEXT_TIME'].agg(
        {'NEXT_TIME_MEAN_DAY':np.mean,'NEXT_TIME_STD_DAY':np.std,'NEXT_TIME_MIN_DAY':np.min,'NEXT_TIME_MAX_DAY':np.max})
    # agg 对不同的字段应用不同的统计逻辑
    tm_feature=pd.merge(tm_feature,l2w,on=['USRID'],how='left')
    tm_feature.fillna(0,inplace=True)
    click_usr=actions[['USRID','DAY']].groupby('USRID').count().reset_index()
    click_usr.columns=['USRID','CLICK_USR']
    tm_feature=pd.merge(tm_feature,click_usr,on=['USRID'],how='left')
    tm_feature['CLICK_USR'].fillna(0,inplace=True)
    # 最后一天点击数占比
    tm_feature['LD_CLICK_RATE']=tm_feature['LD_CLICK']/tm_feature.CLICK_USR
    tm_feature.fillna(-1,inplace=True)
    del tm_feature['CLICK_USR']
    return tm_feature

#模块特征
def module_feature(actions,user):
    mdl_feature=pd.DataFrame()
    mdl_feature['USRID']=user
    #统计每一个模块的点击次数
    mdl=actions['EVT_LBL'].apply(lambda x:x.split('-')) # EVT_LBL 被拆成元组，mdl 是元组组成的数组
    mdl0=pd.DataFrame()
    mdl0['USRID']=actions['USRID']
    mdl0['EVT_LBL']=mdl
    mdl0['F1']=mdl0['EVT_LBL'].apply(lambda x:x[0])
    mdl0['F2']=mdl0['EVT_LBL'].apply(lambda x:x[1])
    mdl0['F3']=mdl0['EVT_LBL'].apply(lambda x:x[2])
    mdl1=mdl0.groupby('USRID')['F1'].values_count().unstack().reset_index()
    # F1应该是第一层级模块的各编号，value_counts()算出每个用户在各编号模块的访问次数，unstack展开后每个模块编号作为字段
    mdl1.fillna(0,inplace=True)
    mdl_feature=pd.merge(mdl_feature,mdl1,on='USRID',how='left')
    f1cnt= mdl0.groupby('USRID')['F1'].apply(lambda x:len(set(x))).reset_index()
    f1cnt.columns=['USRID','F1_CNT']
    f2cnt= mdl0.groupby('USRID')['F2'].apply(lambda x:len(set(x))).reset_index()
    f2cnt.columns=['USRID','F2_CNT']
    f3cnt= mdl0.groupby('USRID')['F3'].apply(lambda x:len(set(x))).reset_index()
    f3cnt.columns=['USRID','F3_CNT']
    mfl_feature=pd.merge(mdl_feature,f1cnt,on='USRID',how='left')
    mfl_feature=pd.merge(mdl_feature,f2cnt,on='USRID',how='left')
    mfl_feature=pd.merge(mdl_feature,f3cnt,on='USRID',how='left')
    mfl_feature.fillna(0,inplace=True)
    topmdl0=actions['EVT_LBL'].value_counts().reset_index()
    topmdl0=actions['EVT_LBL'].value_counts().reset_index()
    topmdl1=topmdl0['INDEX'][topmdl0['EVT_LBL']>=10000]
    topmdl=actions.groupby('USRID')['EVT_LBL'].value_counts().unstack().reset_index()
    topmdl.fillna(0,inplace=True)
    mfl_feature=pd.merge(mdl_feature,topmdl[['USRID']+list(topmdl1)] ,on='USRID',how='left')
    mfl_feature.fillna(0,inplace=True)
    #统计最后5天每个一级模块被访问的次数
    actions=actions[actions['DAY'].isin(['2018-03-28','2018-03-29','2018-03-30','2018-03-31'])]
    l5d_mdl=actions['EVT_LBL'].apply(lambda x:x.split('-')) # EVT_LBL 被拆成元组，mdl 是元组组成的数组
    l5d_mdl0=pd.DataFrame()
    l5d_mdl0['USRID']=actions['USRID']
    l5d_mdl0['EVT_LBL']=l5d_mdl
    l5d_mdl0['F1']=l5d_mdl0['EVT_LBL'].apply(lambda x:x[0])
    l5d_mdl0['F2']=l5d_mdl0['EVT_LBL'].apply(lambda x:x[1])
    l5d_mdl0['F3']=l5d_mdl0['EVT_LBL'].apply(lambda x:x[2])
    l5d_mdl1=mdl0.groupby('USRID')['F1'].values_count().unstack().reset_index() # F1应该是第一层级模块的各编号，value_counts()算出每个用户在各编号模块的访问次数，unstack展开后每个模块编号作为字段
    l5d_mdl1.fillna(0,inplace=True)
    mdl_feature=pd.merge(mdl_feature,l5d_mdl1,on='USRID',how='left')
    # app
    APP=pd.DataFrame()
    APP['USRID']=list(set(actions['USRID']))
    APP['APP']=1
    mdl_feature=pd.merge(mdl_feature,APP,on='USRID',how='left')
    mdl_feature.fillna(0,inplace=True)
    return mdl_feature




