# coding:utf-8
import pandas as pd
import numpy as np
import os
import shutil
import multiprocessing
import time
import gevent
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

# 对数据进行winsorize, neutralize, standardize,
# 处理后的文件存储在 raw_data/after_prehandle.csv
# 该段代码用了多进程代码(148)行，占用内存多，
# 读者根据自己运行环境进行取舍，如果内存过小的用户可以改成循环。


######################################### 通用变量设置 #########################################
start_time = time.time()
raw_data_dir = "./raw_data"

pre_handle_dir = "./pre_handle_data"  # 存放中间数据
if not os.path.exists(pre_handle_dir):
    os.mkdir(pre_handle_dir)

# 申万一级行业分类
sw_map_frame = DataAPI.EquIndustryGet(industryVersionCD=u"010303", industry=u"", secID=u"", ticker=u"", intoDate=u"",
                                      field=[u'ticker', 'secShortName', 'industry', 'intoDate', 'outDate',
                                             'industryName1', 'industryName2', 'industryName3', 'isNew'], pandas="1")
sw_map_frame = sw_map_frame[sw_map_frame.isNew == 1]

# 读入原始因子
input_frame = pd.read_csv(os.path.join(raw_data_dir, u'factor_chpct.csv'),
                          dtype={"ticker": np.str, "tradeDate": np.str, "next_month_end": np.str}, index_col=0)

# 得到因子名
extra_list = ['ticker', 'tradeDate', 'next_month_end', 'abs_return', 'active_return']
factor_name = [x for x in input_frame.columns if x not in extra_list]

print('init data done, cost time: %s seconds' % (time.time() - start_time))


################################### 定义数据处理的一些基本函数 ##################################

def paper_winsorize(v, upper, lower):
    '''
    winsorize去极值，给定上下界
    参数:
        v: Series, 因子值
        upper: 上界值
        lower: 下界值
    返回:
        Series, 规定上下界后因子值
    '''
    if v > upper:
        v = upper
    elif v < lower:
        v = lower
    return v


def winsorize_by_date(cdate_input):
    '''
    按照[dm+5*dm1, dm-5*dm1]进行winsorize
    参数:
        cdate_input: 某一期的因子值的dataframe
    返回:
        DataFrame, 去极值后的因子值
    '''
    media_v = cdate_input.median()
    for a_factor in factor_name:
        dm = media_v[a_factor]
        new_factor_series = abs(cdate_input[a_factor] - dm)  # abs(di-dm)
        dm1 = new_factor_series.median()
        upper = dm + 5 * dm1
        lower = dm - 5 * dm1
        cdate_input[a_factor] = cdate_input[a_factor].apply(lambda x: paper_winsorize(x, upper, lower))
    return cdate_input


def nafill_by_sw1(cdate_input):
    '''
    用申万一级的均值进行填充
    参数:
        cdate_input: 因子值，DataFrame
    返回:
        DataFrame, 填充缺失值后的因子值
    '''
    func_input = cdate_input.copy()
    func_input = func_input.merge(sw_map_frame[['ticker', 'industryName1']], on=['ticker'], how='left')

    func_input.loc[:, factor_name] = func_input.loc[:, factor_name].fillna(
        func_input.groupby('industryName1')[factor_name].transform("mean"))

    return func_input.fillna(0.0)


def winsorize_fillna_date(tdate):
    '''
    对某一天的数据进行去极值，填充缺失值
    参数:
        tdate： 时间， 格式为 %Y%m%d
    返回:
        DataFrame, 去极值，填充缺失值后的因子值
    '''
    cnt = 0
    while True:
        try:
            cdate_input = input_frame[input_frame.tradeDate == tdate]
            # print("####Running single_date for %s" % tdate)
            # winsorize
            cdate_input = winsorize_by_date(cdate_input)

            # 缺失值填充, 用同行业的均值
            cdate_input = nafill_by_sw1(cdate_input)
            cdate_input.set_index('ticker', inplace=True)

            return cdate_input
        except Exception as e:
            cnt += 1
            if cnt >= 3:
                cdate_input = input_frame[input_frame.tradeDate == tdate]
                # 缺失值填充, 用同行业的均值
                cdate_input = nafill_by_sw1(cdate_input)
                cdate_input.set_index('ticker', inplace=True)
                return cdate_input


def standardize_neutralize_factor(input_data):
    '''
    行业、市值中性化，并进行标准化
    参数:
        input_data：tuple, 传入的是(因子值，时间)。因子值为DataFrame
    返回:
        DataFrame, 行业、市值中性化，并进行标准化后的因子值
    '''
    cdate_input, tdate = input_data
    for a_factor in factor_name:
        cnt = 0
        while True:
            try:
                cdate_input.loc[:, a_factor] = standardize(neutralize(cdate_input[a_factor], target_date=tdate,
                                                                      exclude_style_list=['BETA', 'RESVOL', 'MOMENTUM',
                                                                                          'EARNYILD', 'BTOP', 'GROWTH',
                                                                                          'LEVERAGE', 'LIQUIDTY']))
                break
            except Exception as e:
                cnt += 1
                if cnt >= 3:
                    break

    return cdate_input


if __name__ == "__main__":
    ############################################ 对每期的数据进行处理 ###########################################
    # 遍历每个月末日期，对因子进行去极值、空值填充
    print('winsorize factor data...')
    pool = Pool(processes=8)
    date_list = [tdate for tdate in np.unique(input_frame.tradeDate.values) if int(tdate) > 20061231]
    dframe_list = pool.map(winsorize_fillna_date, date_list)

    # 遍历每个月末日期，利用协程对因子进行标准化，中性化处理
    print('standardize & neutralize factor...')
    jobs = [gevent.spawn(standardize_neutralize_factor, value) for value in zip(dframe_list, date_list)]
    gevent.joinall(jobs)
    new_dframe_list = [e.value for e in jobs]
    print('standardize neutralize factor finished!')

    # 将不同月份的数据合并到一起
    all_frame = pd.concat(new_dframe_list, axis=0)
    all_frame.reset_index(inplace=True)

    # 存储下来
    all_frame.to_csv(os.path.join(raw_data_dir, "after_prehandle.csv"), encoding='gbk', chunksize=1000)
    end_time = time.time()
    print("\nData handle finished! Time Cost:%s seconds" % (end_time - start_time))