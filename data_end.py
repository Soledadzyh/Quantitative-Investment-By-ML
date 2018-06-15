import pandas as pd
import numpy as np
import os
import time

start_time = time.time()
raw_data_dir = "./raw_data"

# 给原始数据打上标签，在每个月末截面期，
# 选取下月收益排名前30 % 的股票作为正例（𝑦 = 1），
# 后30 % 的股票作为负例（𝑦 =−1），其余的股票标签为0.
# 处理后的文件存储在
# raw_data / dataset.csv, 文件的数据格式如下：

def get_label_by_return(filename):
    '''
    对下期收益打标签，涨幅top 30%为+1，跌幅top 30%为-1
    参数:
        filename：csv文件名，为上诉步骤保存的因子值
    返回:
        DataFrame, 打完标签后的数据
    '''
    df = pd.read_csv(filename, dtype={"ticker": np.str, "tradeDate": np.str, "next_month_end": np.str}, index_col=0,
                     encoding='gb2312').fillna(0.0)

    new_df = None
    for date, group in df.groupby('tradeDate'):
        quantile_30 = group['active_return'].quantile(0.3)
        quantile_70 = group['active_return'].quantile(0.7)

        def _get_label(x):
            if x >= quantile_70:
                return 1
            elif x <= quantile_30:
                return -1
            else:
                return 0

        group.loc[:, 'label'] = group.loc[:, 'active_return'].apply(lambda x: _get_label(x))

        if new_df is None:
            new_df = group
        else:
            new_df = pd.concat([new_df, group], ignore_index=True)

    return new_df


new_df = get_label_by_return(os.path.join(raw_data_dir, "after_prehandle.csv"))
new_df['year'] = new_df['next_month_end'].apply(lambda x: int(int(x) / 10000))
new_df.to_csv(os.path.join(raw_data_dir, "dataset.csv"), encoding='gbk', chucksize=1000)

print("Done, Time Cost:%s seconds" % (time.time() - start_time))
