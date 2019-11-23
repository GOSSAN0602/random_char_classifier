import pandas as pd
import numpy as np
import sys
sys.path.append("./")
from libs.feature_utils import mk_basic_feature
from libs.data_loader import data_loader
import dask
from dask import dataframe
import multiprocessing

data_dir = "../dataset/"
train_df, _ = data_loader(data_dir)
ddf = dd.from_pandas(train_df,npartitions=2)

meta = train_df.head(1).apply(mk_basic_feature)
res = ddf.apply(mk_basic_feature,meta=meta)
rtn = res.compute(scheduler='process')

#test_df = mk_basic_feature(test_df)
import pdb;pdb.set_trace()
df.to_csv(data_dir+"basic/train_df.csv",index=False)
#pd.to_csv(data_dir+"basic/test_df.csv",index=False)
