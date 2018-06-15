import pandas as pd
import numpy as np


# 定义因子
factors = [b'ChaikinOscillator', b'LCAP', b'REVS20', b'Aroon', b'Hurst']
# Aroon(动量因子)，Hurst(赫斯特指数, 技术指标类因子), ChaikinOscillator(佳庆指标, 技术指标类因子), LCAP(对数市值), REVS20(动量类因子)
# REC，Beta60，PLRC6，DASREV，REVS5

df = pd.read_csv(u'./raw_data/dataset.csv', dtype={"ticker": np.str, "tradeDate": np.str, "next_month_end": np.str},
                 index_col=0, encoding='GBK')
df.head()


class BoostModel:
    def __init__(self, max_depth=3, subsample=0.95, num_round=2000, early_stopping_rounds=50):
        self.params = {'max_depth': max_depth, 'eta': 0.1, 'silent': 1, 'alpha': 0.5, 'lambda': 0.5,
                       'eval_metric': 'auc', 'subsample': subsample, 'objective': 'binary:logistic'}
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, train_data, train_label, val_data, val_label):
        dtrain = xgb.DMatrix(train_data, label=train_label)
        deval = xgb.DMatrix(val_data, label=val_label)

        boost_model = xgb.train(self.params, dtrain, num_boost_round=self.num_round,
                                evals=[(dtrain, 'train'), (deval, 'eval')],
                                early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False)
        print('get best eval auc : %s, in step %s' % (boost_model.best_score, boost_model.best_iteration))
        self.boost_model = boost_model

        return boost_model

    def predict(self, test_data):
        dtest = xgb.DMatrix(test_data)
        predict_score = self.boost_model.predict(dtest, ntree_limit=self.boost_model.best_ntree_limit)

        return predict_score


def get_train_val_test_data(year, split_pct=0.9):
    back_year = max(2007, year - 6)
    train_val_df = df[(df['year'] >= back_year) & (df['year'] < year)]
    train_val_df = train_val_df.sample(frac=1).reset_index(drop=True)

    # 拆分训练集、验证集
    train_df = train_val_df.iloc[0:int(len(train_val_df) * split_pct)]
    val_df = train_val_df.iloc[int(len(train_val_df) * split_pct):]

    test_df = df[df['year'] == year]

    return train_df, val_df, test_df


def format_feature_label(origin_df, is_filter=True):
    if is_filter:
        origin_df = origin_df[origin_df['label'] != 0]
        # 因子xgboost的label输入范围只能是[0, 1]，需要对原始label进行替换
        origin_df['label'] = origin_df['label'].replace(-1, 0)

    feature = np.array(origin_df[factors])
    label = np.array(origin_df['label'])

    return feature, label


def write_factor_to_csv(df, predict_score, year):
    # 记录模型预测分数为因子值，输出
    df['factor'] = predict_score
    df = df.loc[:, ['ticker', 'tradeDate', 'label', 'factor']]
    is_header = True
    if year != 2011:
        is_header = False

    df.to_csv('./raw_data/factor.csv', mode='a+', encoding='utf-8', header=is_header)


def pipeline():
    boost_model_list = []
    for year in range(2011, 2018):
        print('training model for %s' % year)
        train_df, val_df, test_df = get_train_val_test_data(year)
        boost_model = BoostModel()
        train_feature, train_label = format_feature_label(train_df)
        val_feature, val_label = format_feature_label(val_df)

        boost_model.fit(train_feature, train_label, val_feature, val_label)

        test_feature, test_label = format_feature_label(test_df, False)
        predict_score = boost_model.predict(test_feature)

        write_factor_to_csv(test_df, predict_score, year)
        boost_model_list.append(boost_model)

    return boost_model_list


boost_model_list = pipeline()
