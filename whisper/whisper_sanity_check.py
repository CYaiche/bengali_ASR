
import os
import torch
import torchaudio
import pandas as pd
import jiwer 

from whisper.WhisperCustomDataset import WhisperCustomDataset, whisper_collate_fn
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration

model_path          = "bangla-speech-processing/BanglaASR"
feature_extractor   = WhisperFeatureExtractor.from_pretrained(model_path) # MFCC 
tokenizer           = WhisperTokenizer.from_pretrained(model_path)

# Test tokenizer  :
df_train = pd.read_csv("/home/nxp66145/clara/whisper_train.csv")
input_str = df_train.iloc[0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")
