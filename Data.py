# # coding: utf-8
#
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# data = pd.read_csv("flights.csv")
#
# data = data.sample(frac = 0.1, random_state=10)
#
# data = data[["MONTH","DAY","DAY_OF_WEEK","AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT",
#
# "ORIGIN_AIRPORT","AIR_TIME", "DEPARTURE_TIME","DISTANCE","ARRIVAL_DELAY"]]
#
# data.dropna(inplace=True)
#
# data["ARRIVAL_DELAY"] = (data["ARRIVAL_DELAY"]>10)*1
#
# cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
#
# for item in cols:
#
# data[item] = data[item].astype("category").cat.codes +1
#
# train, test, y_train, y_test = train_test_split(data.drop(["ARRIVAL_DELAY"], axis=1), data["ARRIVAL_DELAY"],
#
# random_state=10, test_size=0.25)
from calendar import Calendar

import DataAPI
import pandas as pd
import numpy as np
import os
import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from datetime import date, datetime

from tensorflow import timestamp

raw_data_dir = "./raw_data"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)

# 定义因子
factors = [b'ChaikinOscillator', b'LCAP', b'REVS20', b'Aroon',b'Hurst']
# Aroon(动量因子)，Hurst(赫斯特指数, 技术指标类因子), ChaikinOscillator(佳庆指标, 技术指标类因子), LCAP(对数市值), REVS20(动量类因子)
# REC，Beta60，PLRC6，DASREV，REVS5
def get_factor_by_day(tdate):
    '''
    获取给定日期的因子信息
    参数：
        tdate, 时间，格式%Y%m%d
    返回:
        DataFrame, 返回给定日期的70个因子值
    '''
    cnt = 0
    while True:
        try:
            x = DataAPI.MktStockFactorsOneDayProGet(tradeDate=tdate, secID=u"", ticker=u"",
                                                    field=['ticker', 'tradeDate'] + factors, pandas="1")
            x['tradeDate'] = x['tradeDate'].apply(lambda x: x.replace("-", ""))

            return x
        except Exception as e:
            cnt += 1
            if cnt >= 3:
                print('error get factor data: ', tdate)
                break

if __name__ == "__main__":
    start_time = time.time()

    # 拿到交易日历，得到月末日期
    trade_date = get_trading_dates(start_date='20170101', end_date='20171231')

    c = {"date": trade_date}
    trade_date = pd.DataFrame(c)
    trade_date['date'] = pd.to_datetime(trade_date['date'])
    trade_date.index = trade_date['date']
    dfg = trade_date.groupby(pd.TimeGrouper('M'))
    print(trade_date)
    business_end_day = dfg.agg({'date': np.max})['date'].tolist()
    trade_date=list()
    for i in business_end_day:
          trade_date.append(i._date_repr)
    print(trade_date)
    print("begin to get factor value for each stock...")
#     trade_date = trade_date[trade_date.isMonthEnd == 1]

    print("begin to get factor value for each stock...")
    # # 取得每个月末日期，所有股票的因子值
    pool = ThreadPool(processes=16)
    date_list = [tdate.replace("-", "") for tdate in trade_date if tdate < "20171201"]
    frame_list = pool.map(get_factor_by_day, date_list)
    pool.close()
    pool.join()
    print
    "ALL FINISHED"

    factor_csv = pd.concat(frame_list, axis=0)
    factor_csv.reset_index(inplace=True, drop=True)
    stock_list = np.unique(factor_csv.ticker.values)


    ########################## 取得个股和指数的行情数据 ################################
    print("\nbegin to get price ratio for stocks and index ...")
    # 个股绝对涨幅
    chgframe = DataAPI.MktEqumAdjGet(secID=u"", ticker=stock_list, monthEndDate=u"", isOpen=u"", beginDate=u"20070131",
                                     endDate=u"20171130", field=['ticker', 'endDate', 'tradeDays', 'chgPct', 'return'],
                                     pandas="1")

    chgframe['endDate'] = chgframe['endDate'].apply(lambda x: x.replace("-", ""))

    # 沪深300指数涨幅
    hs300_chg_frame =get_price('000300.XSHG', start_date='2017-01-01', end_date='2017-12-01', frequency='1d', fields=None, country='cn')
    hs300_chg_frame['endDate'] = hs300_chg_frame['endDate'].apply(lambda x: x.replace("-", ""))
    hs300_chg_frame.head()
    # hs300_chg_frame = DataAPI.MktIdxmGet(beginDate=u"20170101", endDate=u"20170601", indexID=u"000300.ZICN", ticker=u"",
    #                                      field=['ticker', 'endDate', 'chgPct'], pandas="1")
    # hs300_chg_frame['endDate'] = hs300_chg_frame['endDate'].apply(lambda x: x.replace("-", ""))
    # hs300_chg_frame.head()

    # 得到个股的相对收益
    hs300_chg_frame.columns = ['HS300', 'endDate', 'HS300_chgPct']
    pframe = chgframe.merge(hs300_chg_frame, on=['endDate'], how='left')
    pframe['active_return'] = pframe['chgPct'] - pframe['HS300_chgPct']
    pframe = pframe[['ticker', 'endDate', 'return', 'active_return']]
    pframe.rename(columns={"return": "abs_return"}, inplace=True)

    ################################ 对齐数据 ################################
    print("begin to align data ...")
    # 得到月度关系
    month_frame = trade_date[['calendarDate', 'isOpen']]
    month_frame['prev_month_end'] = month_frame['calendarDate'].shift(1)
    month_frame = month_frame[['prev_month_end', 'calendarDate']]
    month_frame.columns = ['month_end', 'next_month_end']
    month_frame.dropna(inplace=True)
    month_frame['month_end'] = month_frame['month_end'].apply(lambda x: x.replace("-", ""))
    month_frame['next_month_end'] = month_frame['next_month_end'].apply(lambda x: x.replace("-", ""))

    # 对齐月度关系
    factor_frame = factor_csv.merge(month_frame, left_on=['tradeDate'], right_on=['month_end'], how='left')

    # 得到个股下个月的涨幅数据
    factor_frame = factor_frame.merge(pframe, left_on=['ticker', 'next_month_end'], right_on=['ticker', 'endDate'])

    del factor_frame['month_end']
    del factor_frame['endDate']

    ################################ 数据存储下来 ################################
    factor_frame.to_csv(os.path.join(raw_data_dir, 'factor_chpct.csv'), chunksize=1000)

    end_time = time.time()
    print
    "Time cost: %s seconds" % (end_time - start_time)