# code origin : https://huggingface.co/blog/fine-tune-whisper
import torchaudio 
import os 
import torch
import json
import numpy as np 
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import  DataLoader
import sys
sys.path.append("/home/nxp66145/clara/bengali_ASR/")
from common.common_params import FS
from typing import Any, Dict, List, Union
import soundfile as sf 
import librosa

class WhisperCustomDataset(Dataset):
    def __init__(self, folder_path, dataframe, whisper_processor=None, feature_extractor=None,tokenizer=None):
        self.folder_path        = folder_path
        self.dataframe          = dataframe
        self.sampling_rate      = FS
        if whisper_processor != None :
            self.feature_extractor = whisper_processor.feature_extractor
            self.tokenizer = whisper_processor.tokenizer
        else : 
            self.feature_extractor = feature_extractor
            self.tokenizer = tokenizer

    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        # this function load a single insctance of the dataset 
        # print(f"idx : {idx}")
        # Audio 
        batch = {}
        audio_id = self.dataframe["id"][idx]
        file_path =  os.path.join(self.folder_path, f"{audio_id}.mp3") 
        speech_array, sampling_rate = torchaudio.load(file_path, format="mp3")
        speech_array = speech_array[0].numpy()
        if self.sampling_rate != sampling_rate :
            speech_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=self.sampling_rate)
       
        batch["input_features"]  = self.feature_extractor(speech_array, sampling_rate=self.sampling_rate, return_tensors="pt").input_features[0]

        # Label
        sentences               = self.dataframe["sentence"][idx]
        # print(sentences)
        batch["labels"]         = self.tokenizer(sentences).input_ids

        return  batch



class DataCollatorSpeechSeq2SeqWithPadding():

    def __init__(self, processor) : 
        self.processor =  processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
if __name__ == "__main__" : 
    from transformers import WhisperProcessor
    from transformers import WhisperForConditionalGeneration
    import pandas as pd

    model_path          = "bangla-speech-processing/BanglaASR"
    whisper_processor           =  WhisperProcessor.from_pretrained(model_path)
    # *************** Prepare dataloaders   *************** # 
    audio_folder_pth         = "/disk3/clara/bengali/train_mp3s"
    df_train    = pd.read_csv("/home/nxp66145/clara/whisper_train.csv")
    df_val      = pd.read_csv("/home/nxp66145/clara/whisper_val.csv")


    train_dataset       = WhisperCustomDataset(audio_folder_pth, df_train, whisper_processor)
    val_dataset         = WhisperCustomDataset(audio_folder_pth, df_val, whisper_processor)
    train_loader        = DataLoader(train_dataset, batch_size=1 )
    validation_loader   = DataLoader(val_dataset, batch_size=1 )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=whisper_processor)

    iterator = iter(train_loader)
    data_test = next(iterator)
    print(data_test)
    
    batch = data_collator(data_test)