import pandas as pd 
import os

whisper_results = pd.read_csv("/home/nxp66145/clara/whisper_result_wer_score.csv")


whisper_results.sort_values(by=["wer"], inplace=True, ascending=False)

print(whisper_results.head(100))

# file = open("/home/nxp66145/clara/bengali_ASR/evaluation/whisper_errors.txt", "w")
# file.write(str(whisper_results.head(50)))
# file.close()


for row in range(10) : 
    print(row)
    file_id = whisper_results.iloc[row]["id"]
    cmd = f"cp /disk3/clara/bengali/train_mp3s/{file_id}.mp3 /home/nxp66145/clara/bengali_ASR/audio_to_listen"
    print(cmd)
    os.system(cmd)
# whisper_results.to_csv("//home/nxp66145/clara/bengali_ASR/evaluation/whisper_result_wer_score_sort_by_worst_wer.csv", index=False)