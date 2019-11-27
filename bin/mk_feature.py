import pandas as pd
import numpy as np
import sys
sys.path.append("./")
from libs.feature_utils import mk_basic_feature
from libs.data_loader import data_loader
import argparse

parser = argparse.ArgumentParser(description='Make Feature')
parser.add_argument("type", type=str, help="training or test")
parser.add_argument("size", type=str, help="large or small")
parser.add_argument("idx", type=int, help="X*50000:(X+1)*50000")
args=parser.parse_args()

print("type: ", args.type)
print("size: ", args.size)
print("idx: ", args.idx)

# load data
data_dir = "../dataset/txt/"+args.type+"-data-"+args.size+".txt"
df = pd.read_table(data_dir, header=None)

if args.type=="training":
    df.columns=["target","txt"]
else:
    df.columns=["txt"]

# make feature
df = df.iloc[args.idx*50000:(args.idx+1)*50000]
df = mk_basic_feature(df)

# save
df.to_csv("../dataset/tmp/"+args.type+"-"+args.size+"-"+str(args.idx)+".csv",index=False)
