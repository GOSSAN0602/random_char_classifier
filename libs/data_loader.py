import pandas as pd
import numpy as np

def data_loader(data_dir):
    train_df = pd.read_table(data_dir+"training-data-small.txt",header=None)
    train_df.columns = ["target","txt"]
    test_df = pd.read_table(data_dir+"test-data-small.txt",header=None)
    test_df.columns=["txt"]
    return train_df, test_df
