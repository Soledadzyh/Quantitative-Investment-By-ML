# 模型结果分析
#
# 上述信息只展示了模型验证集上的效果，现在让我们来查看一下样本外的准确率如何。可以看到7个阶段的平均准确率在57%左右，评价AUC在60%左右。

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt


# 计算二分类模型样本外的ACC与AUC
def get_test_auc_acc():
    df = pd.read_csv('./raw_data/factor.csv')
    # 只查看原有label为+1, -1的数据
    df = df[df['label'] != 0]
    df.loc[:, 'predict'] = df.loc[:, 'factor'].apply(lambda x: 1 if x > 0.5 else -1)

    acc_list = []
    auc_list = []
    for date, group in df.groupby('tradeDate'):
        df_correct = group[group['predict'] == group['label']]
        correct = len(df_correct) * 1.0 / len(group)
        auc = roc_auc_score(np.array(group['label']), np.array(group['factor']))
        acc_list.append([date, correct])
        auc_list.append([date, auc])

    acc_list = sorted(acc_list, key=lambda x: x[0], reverse=False)
    mean_acc = sum([item[1] for item in acc_list]) / len(acc_list)

    auc_list = sorted(auc_list, key=lambda x: x[0], reverse=False)
    mean_auc = sum([item[1] for item in auc_list]) / len(auc_list)

    return acc_list, auc_list, round(mean_acc, 2), round(mean_auc, 2)


def plot_accuracy_curve():
    acc_list, auc_list, mean_acc, mean_auc = get_test_auc_acc()

    plt.plot([datetime.strptime(str(item[0]), '%Y%m%d') for item in acc_list], [item[1] for item in acc_list], '-bo')
    plt.plot([datetime.strptime(str(item[0]), '%Y%m%d') for item in auc_list], [item[1] for item in auc_list], '-ro')

    plt.legend([u"acc curve: mean_acc:%s" % mean_acc, u"auc curve: mean auc:%s" % mean_auc], loc='upper left',
               handlelength=2, handletextpad=0.5, borderpad=0.1)
    plt.ylim((0.3, 0.8))
    plt.show()


plot_accuracy_curve()
