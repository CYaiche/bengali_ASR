import torchaudio 
import os 
import torch
import json
import numpy as np 
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from common.common_params import START_TOKEN, MFCC_N_MELS, MFCC_N_FFT, MFCC_HOP_LENGTH, FS


class CustomAudioDataset(Dataset):
    def __init__(self, folder_path, vocabulary_path, dataframe):
        self.folder_path        = folder_path
        self.dataframe          = dataframe
        self.sampling_rate      = FS

        with open(vocabulary_path) as vocabulary_file:
            char_voc = vocabulary_file.read()
        self.vocabulary_to_id = json.loads(char_voc) # convert to dictionary 

    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        # this function load a single insctance of the dataset 
        print(f"idx : {idx}")
        # Audio 
        audio_id = self.dataframe["id"][idx]
        file_path =  os.path.join(self.folder_path, f"{audio_id}.wav") 
        pcm, sample_rate = torchaudio.load(file_path)

        if self.sampling_rate != sample_rate : 
            pcm16k = torchaudio.functional.resample(pcm, sample_rate, self.sampling_rate)

        # Label
        sentence = self.dataframe["sentence"][idx]
        
        # tokenize bengali characters 
        sentence_split = [START_TOKEN] + [ char if len(char) > 0 else _ for char in sentence.replace('',' ').split()]
        sentence_tokenized = [ self.vocabulary_to_id[char] for char in sentence_split ]
        
        return  pcm16k, sentence_tokenized 
