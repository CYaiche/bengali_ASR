import pandas as pd


train_df = pd.read_csv("/home/nxp66145/clara/train.csv")
print(train_df.shape)
test_id = pd.read_csv("/home/nxp66145/clara/bengali_ASR/save_test_id_for_baseline.csv")
print(test_id.shape)

test_df = pd.DataFrame(columns =train_df.columns)

for row_idx in range(test_id.shape[0]) : 
    print(row_idx)
    file_id = test_id.iloc[row_idx]["id"]
    # print(file_id)
    
    # print(train_df[train_df["id"] == file_id].index)
    row = pd.DataFrame({"id": file_id,
                        "sentence": train_df[train_df["id"] == file_id]["sentence"].values[0],
                        "split" : "train"}, index=[0])
    # print(row.head())
    test_df = pd.concat([test_df, row], ignore_index=True)
    train_df.drop(train_df[train_df["id"] == file_id].index, inplace=True)



# print(train_df.shape)
# print(test_df.shape)
out_train_path = "/home/nxp66145/clara/whisper_train.csv"
train_df.to_csv(out_train_path)

out_test_path = "/home/nxp66145/clara/whisper_test.csv"
test_df.to_csv(out_test_path)


