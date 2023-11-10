import os, json
import pandas as pd 
import lightning.pytorch as pl
import torchaudio 
import torch 
from torch.utils.data import  DataLoader
from data_preparation.CustomAudioDataset import CustomAudioDataset
from common.common_params import FS, START_TOKEN, MFCC_N, MFCC_N_FFT, MFCC_HOP_LENGTH, BATCH_SIZE
from sklearn.model_selection import train_test_split

    
class RNNTDataModule(pl.LightningDataModule):
    def __init__(self, folder_path, vocabulary_path, metadata_path, max_label_size, batch_size=BATCH_SIZE, max_input_length=None):
        super().__init__()
        self.folder_path        = folder_path
        self.metadata_path      = metadata_path
        self.max_input_length   = max_input_length # in seconds
        self.max_label_size     = max_label_size 
        self.vocabulary_path    = vocabulary_path
        self.batch_size         = BATCH_SIZE
        
        self.mfcc   =  torchaudio.transforms.MFCC(sample_rate=FS,       
                                n_mfcc=MFCC_N,    # Number of MFCC coefficients
                                # 25ms FFT window  10ms frame shift
                                melkwargs={'n_fft': MFCC_N_FFT, 'hop_length': MFCC_HOP_LENGTH}  
                        )

    def prepare_data(self):
        # download
        self.dataframe =  pd.read_csv(self.metadata_path)
        self.train_dataframe, self.val_dataframe = train_test_split(self.dataframe, test_size=0.2,shuffle=True)
        self.train_dataframe.reset_index(inplace=True, drop=True)
        self.val_dataframe.reset_index(inplace=True, drop=True)
        print(self.dataframe.columns)
        print(f"nb train pairs :{self.train_dataframe.shape}")
        print(f"nb validation pairs :{self.val_dataframe.shape}")
        print(self.train_dataframe.columns)
        self.train_dataset = CustomAudioDataset(self.folder_path,
                                        self.vocabulary_path,
                                        self.train_dataframe)
        
        self.val_dataset = CustomAudioDataset(self.folder_path,
                                        self.vocabulary_path,
                                        self.val_dataframe)
    
    def collate_fn(self, batch):
        batch_size = len(batch)
        T = [ pcm16k.shape[1]  for (pcm16k,_) in batch]
        U = [ len(sentence_tokenized) for (_,sentence_tokenized) in batch]
        max_t = max(T)
        max_u = max(U)
        spectrograms = []
        sentence_tokenized_pad_s =[]

        for index in range(batch_size) : 
            pcm16k, sentence_tokenized = batch[index]
            # pad audio and mffcc
            pad_right = int( max_t - pcm16k.shape[1])
            assert pad_right >=0 , "error , one audio file superior to max_input_length 16k"
            pcm16kpad = torch.nn.functional.pad(pcm16k, (0, pad_right) , "constant", 0)
            spectrogram = self.mfcc(pcm16kpad)
            # pad tokens 
            pad_right = int(max_u  - len(sentence_tokenized))
            sentence_tokenized_pad = torch.nn.functional.pad(torch.tensor(sentence_tokenized), (0, pad_right) , "constant", 0)

            spectrograms.append(spectrogram)
            sentence_tokenized_pad_s.append(sentence_tokenized_pad)
            
        spectrograms = torch.stack(spectrograms)
        sentence_tokenized_pad_s = torch.stack(sentence_tokenized_pad_s)
        T_f = [int(t / MFCC_HOP_LENGTH) +1  for t in T] # becuse of mfcc padding
        return (spectrograms, sentence_tokenized_pad_s, torch.tensor(T_f),  torch.tensor(U)) 
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                        collate_fn=self.collate_fn,
                        batch_size=self.batch_size,
                        num_workers=4,
                        # shuffle=True
                        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                        collate_fn=self.collate_fn,
                        batch_size=self.batch_size,
                        num_workers=4,
                        # shuffle=True
                        )

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=32)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=32)