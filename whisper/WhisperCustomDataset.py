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


class WhisperCustomDataset(Dataset):
    def __init__(self, folder_path, dataframe, whisper_processor):
        self.folder_path        = folder_path
        self.dataframe          = dataframe
        self.sampling_rate      = FS
        self.whisper_processor =  whisper_processor

    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        # this function load a single insctance of the dataset 
        # print(f"idx : {idx}")
        # Audio 
        audio_id = self.dataframe["id"][idx]
        file_path =  os.path.join(self.folder_path, f"{audio_id}.mp3") 
        pcm, sample_rate = torchaudio.load(file_path)

        if self.sampling_rate != sample_rate : 
            pcm16k = torchaudio.functional.resample(pcm, sample_rate, self.sampling_rate)
        batch = {}
        batch["input_features"] = self.whisper_processor.feature_extractor(pcm16k[0,:], sampling_rate= self.sampling_rate).input_features[0]

        # Label
        sentences               = self.dataframe["sentence"][idx]
        batch["labels"]         = self.whisper_processor.tokenizer(sentences).input_ids

        return  batch


class DataCollatorSpeechSeq2SeqWithPadding():
    def __init__(self, processor) : 
        self.whisper_processor =  processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        audios = [{"input_features" : batch["input_features"] } for batch in features]
        sentences = [{"input_ids" : batch["labels"] } for batch in features]


        batch      = self.whisper_processor.feature_extractor.pad(audios, return_tensors="pt")
        tokenized_sentences_pad = self.whisper_processor.tokenizer.pad(sentences, return_tensors="pt")

        # Transformer models is attention layers that contextualize each token. These will
        # take into account the padding tokens since they attend to all of the tokens of a sequence.
        # To get the same result when passing individual sentences of different lengths through 
        # the model or when passing a batch with the same sentences and padding applied, 
        # we need to tell those attention layers to ignore the padding tokens. 
        # This is done by using an attention mask.

        # replace padding with -100 to ignore loss correctly
        labels = tokenized_sentences_pad["input_ids"].masked_fill(tokenized_sentences_pad.attention_mask.ne(1), -100)
        
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.whisper_processor.tokenizer.bos_token_id).all().cpu().item():
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
    df_val      = pd.read_csv("/home/nxp66145/clara/whisper_test.csv")


    train_dataset       = WhisperCustomDataset(audio_folder_pth, df_train, whisper_processor)
    val_dataset         = WhisperCustomDataset(audio_folder_pth, df_val, whisper_processor)
    train_loader        = DataLoader(train_dataset, batch_size=1 )
    validation_loader   = DataLoader(val_dataset, batch_size=1 )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=whisper_processor)

    iterator = iter(train_loader)
    data_test = next(iterator)
    print(data_test)
    
    batch = data_collator(data_test)