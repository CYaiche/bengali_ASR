import pandas as pd
from bltk.langtools.banglachars import punctuations
# Error is training might code for unclean transcript 

def preprocessing():
    # remove punctuation 

df_train                 = pd.read_csv("/home/nxp66145/clara/whisper_train_sample_100.csv",index_col=0)
df_val                 = pd.read_csv("/home/nxp66145/clara/whisper_val_sample_100.csv",index_col=0)








df_train.to_csv("/home/nxp66145/clara/whisper_train_sample_100.csv")
df_val.to_csv("/home/nxp66145/clara/whisper_val_sample_100.csv")
