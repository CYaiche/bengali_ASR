import os
from typing import List, Optional, Union, Dict

import tqdm
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Wav2Vec2Processor
)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: List,
        basedir: Optional[str] = None,
        sampling_rate: int = 16000,
        max_audio_len: int = 5,
    ):
        self.dataset = dataset
        self.basedir = basedir

        self.sampling_rate = sampling_rate
        self.max_audio_len = max_audio_len

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.dataset)

    def _cutorpad(self, audio: np.ndarray) -> np.ndarray:
        """
        Cut or pad audio to the wished length
        """
        effective_length = self.sampling_rate * self.max_audio_len
        len_audio = len(audio)

        # If audio length is bigger than wished audio length
        if len_audio > effective_length:
            audio = audio[:effective_length]

        # Expand one dimension related to the channel dimension
        return audio


    def __getitem__(self, index) -> torch.Tensor:
        """
        Return the audio and the sampling rate
        """
        if self.basedir is None:
            filepath = self.dataset[index]
        else:
            filepath = os.path.join(self.basedir, self.dataset[index])

        speech_array, sr = torchaudio.load(filepath)

        # Transform to mono
        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

        if sr != self.sampling_rate:
            transform = torchaudio.transforms.Resample(sr, self.sampling_rate)
            speech_array = transform(speech_array)
            sr = self.sampling_rate

        speech_array = speech_array.squeeze().numpy()

        # Cut or pad audio
        speech_array = self._cutorpad(speech_array)

        return speech_array

class CollateFunc:
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        pad_to_multiple_of: Optional[int] = None,
        sampling_rate: int = 16000,
    ):
        self.padding = padding
        self.processor = processor
        self.max_length = max_length
        self.sampling_rate = sampling_rate
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List):
        input_features = []

        for audio in batch:
            input_tensor = self.processor(audio, sampling_rate=self.sampling_rate).input_values
            input_tensor = np.squeeze(input_tensor)
            input_features.append({"input_values": input_tensor})

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return batch


def predict(test_dataloader, model, device: torch.device):
    """
    Predict the class of the audio
    """
    model.to(device)
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader):
            input_values, attention_mask = batch['input_values'].to(device), batch['attention_mask'].to(device)

            logits = model(input_values, attention_mask=attention_mask).logits
            scores = F.softmax(logits, dim=-1)

            pred = torch.argmax(scores, dim=1).cpu().detach().numpy()

            preds.extend(pred)

    return preds


def get_gender(model_name_or_path: str, audio_paths: List[str], label2id: Dict, id2label: Dict, device: torch.device):
    num_labels = 2

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
    model = AutoModelForAudioClassification.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    test_dataset = CustomDataset(audio_paths)
    data_collator = CollateFunc(
        processor=feature_extractor,
        padding=True,
        sampling_rate=16000,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=10
    )

    preds = predict(test_dataloader=test_dataloader, model=model, device=device)

    return preds


def get_gender_list(audio_paths) : 
    model_name_or_path = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label2id = {
        "female": 0,
        "male": 1
    }

    id2label = {
        0: "female",
        1: "male"
    }

    num_labels = 2

    preds = get_gender(model_name_or_path, audio_paths, label2id, id2label, device)
    
    return preds 
if __name__ == "__main__" : 
   
    audio_paths = ["/disk3/clara/bengali/train_mp3s/750d6a12fe1c.mp3",
                   "/disk3/clara/bengali/train_mp3s/3fc278c49e7a.mp3",
                   "/disk3/clara/bengali/train_mp3s/f76fe07c6f5b.mp3"] # Must be a list with absolute paths of the audios that will be used in inference
    preds = get_gender_list(audio_paths) 
    print(preds)
