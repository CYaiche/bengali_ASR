import torchaudio 
import os 
import torch
import json
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader

class CustomAudioDataset(Dataset):
    def __init__(self, folder_path, vocabulary_path, dataframe, sampling_rate, max_label_size, max_input_length=None):
        self.folder_path        = folder_path
        self.dataframe          = dataframe
        self.sampling_rate      = sampling_rate
        self.max_input_length   = max_input_length # in seconds
        self.max_label_size     = max_label_size
        n_mel = 80
        self.mfcc   =  torchaudio.transforms.MFCC(sample_rate=sampling_rate,       
                                        n_mfcc=n_mel,    # Number of MFCC coefficients
                                        # 25ms FFT window  10ms frame shift
                                        melkwargs={'n_fft': 400, 'hop_length': 160}  
                                )
        with open(vocabulary_path) as vocabulary_file:
            char_voc = vocabulary_file.read()
        self.vocabulary_to_id = json.loads(char_voc) # convert to dictionary 
        print(self.vocabulary_to_id)
        
    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        
        # Audio 
        audio_id = self.dataframe["id"][idx]
        file_path =  os.path.join(self.folder_path, f"{audio_id}.wav") 
        pcm, sample_rate = torchaudio.load(file_path)

        if self.sampling_rate != sample_rate : 
            pcm16k = torchaudio.functional.resample(pcm, sample_rate, self.sampling_rate)

        if  self.max_input_length != None : # apply padding
            pad_right = int(( self.max_input_length * self.sampling_rate ) - pcm16k.shape[1])
            assert pad_right >=0 , "error , one audio file superior to max_input_length 16k"
            pcm16kpad = torch.nn.functional.pad(pcm16k, (0, pad_right) , "constant", 0)

        spectrograms = self.mfcc(pcm16kpad)
        
        # Label
        
        label = self.dataframe["sentence"][idx]
        label_encoded = torch.tensor([ self.vocabulary_to_id[char] for char in list(label)])
        pad_right = int(self.max_label_size  - label_encoded.shape[0])
        label_encoded_pad = torch.nn.functional.pad(label_encoded, (0,pad_right), "constant", 0)
        return spectrograms, label_encoded_pad