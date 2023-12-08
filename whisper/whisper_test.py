
from pathlib import Path
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
import matplotlib.pyplot as plt


class WhisperTest():
    def __init__(self, model_path) : 

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # feature extractor from model
        # whisper_base_model = "bangla-speech-processing/BanglaASR"
        whisper_base_model      = "openai/whisper-small"
        self.feature_extractor  = WhisperFeatureExtractor.from_pretrained(model_path)
        # self.processor           = WhisperProcessor.from_pretrained(model_path)

        # load model or fine-tuned model
        
        self.tokenizer           = WhisperTokenizer.from_pretrained(whisper_base_model)
        self.model               = WhisperForConditionalGeneration.from_pretrained(model_path).to(self.device)

        

    def run_inference(self, audio_path) : 
        speech_array, sampling_rate = torchaudio.load(audio_path, format="mp3")
        speech_array = speech_array[0].numpy()
       
        # np.save(f"/home/nxp66145/clara/bengali_ASR/whisper/npy/{audio_path.name.split('.')[0]}.npy", speech_array, allow_pickle=True)

        speech_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=16000)
        input_features = self.feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt").input_features
        # np.save(f"/home/nxp66145/clara/bengali_ASR/whisper/npy/{audio_path.name.split('.')[0]}_spec.npy", input_features, allow_pickle=True)
        predicted_ids = self.model.generate(inputs=input_features.to(self.device))[0]
        transcription = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)

        return transcription




def run_inference_on_data_set(model_path, test_set, csv_out_path, test=False):
    data_loc = Path("/disk3/clara/bengali/train_mp3s/")
    if test : 
        data_loc = Path("/disk3/clara/bengali/test/")
    whisper_test = WhisperTest(model_path)
    submission = pd.DataFrame(columns=["id", "sentence"])

    test_set["predicted"] = test_set.apply(lambda x: whisper_test.run_inference(data_loc/ f"{x.id}.mp3"), axis=1 )
    
    submission =  test_set[["id","sentence","predicted"]]
    print(csv_out_path)
    submission.to_csv(csv_out_path)

if __name__ == "__main__" : 
    root_repo = Path("/home/nxp66145/clara/bengali_ASR/")
    test_gengla_model = True
    if test_gengla_model : 
        model_path = "bangla-speech-processing/BanglaASR"
        
        
        csv_out_path = root_repo / "whisper_sample_100.csv"
    test_openAI = False
    if test_openAI : 
        model_path = "openai/whisper-small"
        csv_out_path = root_repo / "whisper_small_on_val_set.csv"

    df_val                  = pd.read_csv("/home/nxp66145/clara/whisper_train_sample_100.csv")
    run_inference_on_data_set(model_path,df_val, csv_out_path)


