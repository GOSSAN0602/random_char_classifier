import pandas as pd
import numpy as np

def mk_basic_feature(df):
    for i in range(df.shape[0]):
        words = df["txt"].iloc[i].split(",")
        for idx, word in enumerate(words):
            col_name = "word_" + str(idx)
            df.loc[i, col_name] = word
            df.loc[i, "1stchar_of_word_"+str(idx)] = word[0]
            df.loc[i, "num_of_word_"+str(idx)] = int(word[1:])
            df.loc[i, "len_num_of_word_"+str(idx)] = len(word[1:])
        df.loc[i,"n_words"] = len(words)
    return df
    
