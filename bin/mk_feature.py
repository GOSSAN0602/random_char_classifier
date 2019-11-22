import pandas as pd
import numpy as np
from libs.feature_utils import mk_basic_feature
from libs.data_loader import data_loader

data_dir = "../data/"
train_df, test_df = data_loader(data_dir)

train_df = mk_basic_feature(train_df)
test_df = mk_basic_feature(test_df)

pd.to_csv(data_dir+"basic/train_df.csv",index=False)
pd.to_csv(data_dir+"basic/test_df.csv",index=False)
