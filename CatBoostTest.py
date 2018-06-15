from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split

data = pd.read_csv("testxy.csv")
# open,close,high,low,volume,ema,macd,linreg,momentum,rsi,var,cycle,atr,values
data = data[
    ["open", "close", "high", "low", 'volume', 'ema', 'macd', 'linreg', 'momentum', 'rsi', 'var', 'cycle', 'atr',
     'values']]
data.dropna(inplace=True)
data["values"] = (data["values"] == 1)*1
data["rsi"] = (data["rsi"] > 50)*1
data["macd"] = (data["macd"] > 50)*1
data["momentum"] = (data["momentum"] > 0)*1
# rsi:50
# macd:50
# 下根K线涨跌作为预测指标
# momentum:0
cols = ["rsi", "macd", "momentum"]
for item in cols:
    data[item] = data[item].astype("category").cat.codes + 1

train, test, y_train, y_test = train_test_split(data.drop(["values"], axis=1), data["values"],
                                                random_state=0, test_size=0.3)
import catboost as cb

cat_features_index = [5, 6, 7, 8, 9, 10, 11]


def auc(m, train, test):
    return (metrics.roc_auc_score(y_train, m.predict_proba(train)[:, 1]),
            metrics.roc_auc_score(y_test, m.predict_proba(test)[:, 1]))


params = {'depth': [4, 7, 10],
          'learning_rate': [0.03, 0.1, 0.15],
          'l2_leaf_reg': [1, 4, 9],
          'iterations': [300],
          'plot':['True']}

# cb = cb.CatBoostClassifier()
# cb_model = GridSearchCV(cb, params, scoring="roc_auc", cv=3)
# cb_model.fit(train, y_train)

# # With Categorical features
# clf = cb.CatBoostClassifier(eval_metric="AUC", depth=10, iterations=500, l2_leaf_reg=9, learning_rate=0.15)
# clf.fit(train, y_train)
# auc(clf, train, test)

# With Categorical features

clf = cb.CatBoostClassifier(eval_metric="AUC", one_hot_max_size=50,depth=10, iterations=500, l2_leaf_reg=9, learning_rate=0.15)
clf.fit(train, y_train, cat_features=cat_features_index)
auc(clf, train, test)
