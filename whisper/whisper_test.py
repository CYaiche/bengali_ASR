
import os
import librosa
import torch
import torchaudio
import numpy as np
import pandas as pd
import jiwer 
import soundfile as sf
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration

def speech_to_text(audio_path): 
    speech_array, sampling_rate = torchaudio.load(audio_path, format="mp3")
    speech_array = speech_array[0].numpy()
    speech_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=16000)
    input_features = feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt").input_features

    predicted_ids = model.generate(inputs=input_features.to(device))[0]


    transcription = processor.decode(predicted_ids, skip_special_tokens=True)

    return transcription

CUR_DIR =  os.getcwd()
DATA    = os.path.join(CUR_DIR , "bengaliai-speech")
TRAIN   =  os.path.join( DATA , "train_mp3s")
TEST    =  os.path.join(DATA , "test_mp3s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mp3_path = "https://huggingface.co/bangla-speech-processing/BanglaASR/resolve/main/mp3/common_voice_bn_31515636.mp3"

model_path = "bangla-speech-processing/BanglaASR"
root = "/home/nxp66145/clara/bengali_ASR/"

feature_extractor   = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer           = WhisperTokenizer.from_pretrained(model_path)
processor           = WhisperProcessor.from_pretrained(model_path)
model               = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
id_to_test = pd.read_csv(os.path.join(root, "save_test_id_for_baseline.csv"))
id_to_test = id_to_test["id"].values
submission = pd.DataFrame(columns=["id", "sentence"])
i = 0 
for id in id_to_test : 
    audio_path = os.path.join("/disk3/clara/bengali/train_mp3s/",f"{id}.mp3")
    id = os.path.basename(audio_path).split(".")[0]
    text = speech_to_text(audio_path) 
    submission.loc[i] = [id, text]
    i = i + 1

print(submission.columns)
submission.to_csv("whisper.csv")
