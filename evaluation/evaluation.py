import jiwer 
import pandas as pd 
import numpy as np 

def mean_wer(solution, submission):
    joined = solution.merge(submission.rename(columns={'sentence': 'predicted'}))
    domain_scores = joined.groupby('domain').apply(
        # note that jiwer.wer computes a weighted average wer by default when given lists of strings
        lambda df: jiwer.wer(df['sentence'].to_list(), df['predicted'].to_list()),
    )
    return domain_scores.mean()



def write_wer_per_sample(test_df, whisper_output):
    merge = pd.merge(test_df[["id","sentence"]], whisper_output[["id","sentence"]], how="right", on=["id","id"] )
    merge.fillna("", inplace=True)
    merge["wer"]   = merge.apply(lambda x : jiwer.wer(x.sentence_x,x.sentence_y), axis=1)

    print(merge["wer"].mean())
    merge.columns = ["id","true_label","whisper_pred","wer"]
    merge.to_csv("/home/nxp66145/clara/whisper_result_wer_score.csv", index=False)

if __name__ == "__main__" : 

    true_label          = pd.read_csv("/home/nxp66145/clara/whisper_val_sample_100.csv")
    # result_under_eval   = pd.read_csv("/disk3/clara/bengali/whisper_small_val_100.csv")
    result_under_eval   = pd.read_csv("/home/nxp66145/clara/bengali_ASR/whisper_dry_run_whisper_small_missing_checkpoint.csv")
    WER = [] 

    # for row_id in range(result_under_eval.shape[0]) : 
    #     print(row_id)
    #     row = result_under_eval.iloc[row_id]
    #     # print(row["id"])
    #     true_label_corresponding_to_id = true_label[true_label["id"] == row["id"]]["sentence"]
    #     # print(true_label_corresponding_to_id.values[0])
    #     sentence = row["predicted"] if not type(row["predicted"]) == float else ""
    #     print(sentence)
    #     print(true_label_corresponding_to_id.values)
    #     wer_score = jiwer.wer(true_label_corresponding_to_id.values[0], sentence) 
    #     print(wer_score)
    #     WER.append(wer_score)
    # WER = np.array(WER)
    # print("MEAN WER" , np.mean(WER))
    for row_id in range(result_under_eval.shape[0]) : 
        print(row_id)
        row = result_under_eval.iloc[row_id]
        # print(row["id"])
        true = row["sentence"]
        predicted = row["predicted"]

        wer_score = jiwer.wer(true, predicted) 
        print(wer_score)
        WER.append(wer_score)
    WER = np.array(WER)
    print("MEAN WER" , np.mean(WER))
    # WER RESULT PER SAMPLES 
    # write_wer_per_sample(true_label, result_under_eval)