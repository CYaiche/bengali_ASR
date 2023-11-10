import os, sys
import pandas as pd
from common.common_params import EXTRACT_DIR
import librosa 
import soundfile as sf


audio_loc = os.path.join(EXTRACT_DIR, "life")

def get_audio_len(audio_id):
    file_path =  os.path.join(audio_loc, f"{audio_id}.wav") 
    return librosa.get_duration(filename=file_path)

def audio_smapling_rate(audio_id):
    f = sf.SoundFile(os.path.join(audio_loc, f"{audio_id}.wav") )
    return f.samplerate

metadata_path = os.path.join(EXTRACT_DIR, "metadata","life_clean.csv")
metadata_path_out = os.path.join(EXTRACT_DIR, "metadata","life_clean_enh.csv")
metadata_text = os.path.join(EXTRACT_DIR, "metadata","metadata_of_metadata.txt")
metadata = pd.read_csv(metadata_path)

metadata["duration(s)"] = metadata["id"].apply(get_audio_len)
metadata["sr"] = metadata["id"].apply(audio_smapling_rate)
metadata["text_length"] = metadata["sentence"].apply(len)


metadata.drop(columns=["Unnamed: 0.1", "Unnamed: 0"], inplace=True)

# metadata.to_csv(metadata_path_out)


metadata.sort_values("duration(s)", inplace=True)
metadata.to_csv(metadata_path_out)

dic_info = {
    "shape" : metadata.shape,
    "mean_duration" : metadata["duration(s)"].mean(),
    "min_duration" : metadata["duration(s)"].min(),
    "max_duration" : metadata["duration(s)"].max(),
    "mean_sentence_length" : metadata["text_length"].mean(),
    "min_sentence_length" : metadata["text_length"].min(),
    "max_sentence_length" : metadata["text_length"].max()
}
with open(metadata_text, 'w') as f:
    for key in dic_info.keys() :
        f.write(key)
        f.write(" : ")
        f.write(str(dic_info[key]))
        f.write('\n')
