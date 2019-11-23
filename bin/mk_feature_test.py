import pandas as pd
import numpy as np
import sys
sys.path.append("./")
from libs.feature_utils import mk_basic_feature
from libs.data_loader import data_loader
import gc

data_dir = "../dataset/"
_, test_df = data_loader(data_dir)
test_df = mk_basic_feature(test_df)

test_df.to_csv(data_dir+"basic/test_df.csv",index=False)
