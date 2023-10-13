
import os 
from pydub import AudioSegment

def extract_audio_from_word_in_metadata(word, metadata, src_foler, dst_foler, save_csv_path=None): 
    mask = metadata["sentence"].str.contains(word)
    life_audio_metadata = metadata[mask]
    life_audio_metadata.shape
    if save_csv_path != None: 
        life_audio_metadata.to_csv("../extracted/metadata/life.csv")
    
    file_list = [ file_id + ".mp3" for file_id in life_audio_metadata["id"].tolist() ]
    
    for filename in file_list : 
        src = os.path.join(src_foler, filename)
        dst = os.path.join(dst_foler, f"{filename.split('.')[0]}.wav")
        
        # convert wav to mp3                                                            
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")

