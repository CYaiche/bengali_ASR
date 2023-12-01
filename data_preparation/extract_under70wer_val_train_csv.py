import pandas as pd


col = ["id", "ggl_wer", "ykg_wer"]

baseline = pd.read_csv("/home/nxp66145/clara/bengali_ASR/train_metadata.csv")

df_train                 = pd.read_csv("/home/nxp66145/clara/whisper_train.csv", index_col=0)
print(df_train.shape)
merge_train         = pd.merge(baseline[col], df_train[["id", "sentence"]], how="right" ,on=["id","id"])
under70_train       = merge_train[merge_train["ggl_wer"] < 0.7][["id", "sentence","ggl_wer"]]
print(under70_train.shape)


df_val              = pd.read_csv("/home/nxp66145/clara/whisper_val.csv", index_col=0)
print(df_val.shape)
merge_val           = pd.merge(baseline[col], df_val[["id", "sentence"]], how="right" ,on=["id","id"])
under70_val   = merge_val[merge_val["ggl_wer"] < 0.7][["id", "sentence","ggl_wer"]]
print(under70_val.shape)


under70_train.to_csv("/home/nxp66145/clara/whisper_train_under70.csv")
under70_val.to_csv("/home/nxp66145/clara/whisper_val_under70.csv")