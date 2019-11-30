from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('./')
from libs.get_params import get_lgb_params
#from libs.feature_select import kolmogorov_smirnov, adversarial_del_list
import lightgbm as lgb
import numpy as np
import pandas as pd
import tables
import gc

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load Data
data_folder = '../dataset/tmp/'
X = pd.read_csv(data_folder+'training-small-0.csv')
#X_test = pd.read_hdf(data_folder+'test.hdf', "df", engine='python')
for i in X.columns:
    X.loc[X[i]=="X",i]=1
    X.loc[X[i]=="Y",i]=2
    X.loc[X[i]=="Z",i]=3

    
X.drop(["txt"],axis=1,inplace=True)
X.fillna(np.nan,inplace=True)
y = X.loc[:,"target"].copy()
X.drop(["target"],axis=1,inplace=True)

# make feature
basic_f_imp = pd.read_csv("./basic_feature_importances.csv")
top45 = basic_f_imp["feature"][:45].values.tolist()
top45.extend(["cnt_X","cnt_Z"])
for i in top45:
    for j in top45:
        col_name=str(i)+"_"+str(j)+"_"
        X[col_name+"mul"]=X[i]*X[j]
        X[col_name+"div"]=X[i]/X[j]
del basic_f_imp
gc.collect()

# select feature
f_imp=pd.read_csv("feature_importances.csv")
use_col=f_imp["feature"].values[:500].tolist()
X=X.loc[:,use_col]

#del_list=["num_of_word_0","num_of_word_1","num_of_word_2","num_of_word_3"]
del_list=["num_of_word_0_num_of_word_0_mul","num_of_word_1_num_of_word_1_mul"]
X.drop(del_list,axis=1,inplace=True)

# config
NFOLDS = 5
folds = KFold(n_splits=NFOLDS)
params = get_lgb_params()

columns = X.columns
splits = folds.split(X, y)
#y_preds = np.zeros(X_test.shape[0])
y_oof = np.zeros(X.shape[0])
score = 0

feature_importances = pd.DataFrame()
feature_importances['feature'] = columns

# Train
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
 #   y_preds += clf.predict(X_test) / NFOLDS

    del X_train, X_valid, y_train, y_valid
    gc.collect()

print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")

# make submission file
#sub['isFraud'] = y_preds
#sub.to_csv("submission.csv", index=False)

# feature importance fig
feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)
feature_importances.sort_values(by='average', ascending=False, inplace=True)
plt.figure(figsize=(24, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title(str(score))
plt.savefig("feature_importance.png")

# save feature importances
feature_importances.to_csv("./feature_importances.csv")
