
from transformers import WhisperForConditionalGeneration
from pathlib import Path
import pandas as pd 
from whisper_test import run_inference_on_data_set
from whisper.WhisperCustomDataset import WhisperCustomDataset
from transformers import WhisperProcessor


model_id                = "dry_run_test_compute_metric_20231124_141914" # "20231114_180857"
checkpoint = ""
model_path              = f"/disk2/clara/whisper/{model_id}/{checkpoint}"
root_repo               = Path("/home/nxp66145/clara/bengali_ASR/")
whisper_base_model      = "bangla-speech-processing/BanglaASR"
csv_out_path            = root_repo / f"whisper_{model_id}.csv"

audio_folder_pth        = "/disk3/clara/bengali/train_mp3s"

# df_val                  = pd.read_csv("/home/nxp66145/clara/whisper_val.csv")
df_val                  = pd.read_csv("/home/nxp66145/clara/whisper_train_sample_100.csv")

# df_val = df_val.sample(10).reset_index(drop=True)
whisper_processor =  WhisperProcessor.from_pretrained(whisper_base_model)

run_inference_on_data_set(model_path, df_val, csv_out_path)
