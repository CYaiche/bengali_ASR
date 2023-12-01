import pandas as pd


df_train                 = pd.read_csv("/home/nxp66145/clara/whisper_train.csv",index_col=0)
df_val                   = pd.read_csv("/home/nxp66145/clara/whisper_val.csv",index_col=0)

df_train = df_train.sample(100)
df_val = df_val.sample(100)

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)

df_train.to_csv("/home/nxp66145/clara/whisper_train_sample_100.csv")
df_val.to_csv("/home/nxp66145/clara/whisper_val_sample_100.csv")

