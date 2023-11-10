import pandas as pd 


col = ["id", "ggl_wer", "ykg_wer"]
baseline = pd.read_csv("/home/nxp66145/clara/bengali_ASR/train_metadata.csv")
test_df = pd.read_csv("/home/nxp66145/clara/whisper_test.csv")

merge = pd.merge(baseline[col], test_df[["id", "sentence"]], how="right" ,on=["id","id"])


print("Google", merge["ggl_wer"].mean())
print("Yellowking", merge["ykg_wer"].mean())