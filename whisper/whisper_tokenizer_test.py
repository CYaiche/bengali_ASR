from transformers import WhisperProcessor
import pandas as pd
from whisper.WhisperCustomDataset import WhisperCustomDataset
from torch.utils.data import DataLoader

model_path                      = "bangla-speech-processing/BanglaASR"
model_path = "openai/whisper-small"
whisper_processor = WhisperProcessor.from_pretrained(model_path)

audio_folder_pth         = "/disk3/clara/bengali/train_mp3s"
df_val                   = pd.read_csv("/home/nxp66145/clara/whisper_val.csv")

val_dataset     = WhisperCustomDataset(audio_folder_pth, df_val, whisper_processor)


loader = DataLoader(val_dataset, batch_size=1)

# for batch_ndx, sample in enumerate(loader):
#     print(batch_ndx)
#     print(sample)
#     print(sample["input_features"].shape)
#     print(sample["labels"])
#     if batch_ndx > 10 : 
#         break

from bltk.langtools.banglachars import (vowels,
                                            vowel_signs,
                                            consonants,
                                            digits,
                                            operators,
                                            punctuations,
                                      others)


def display_encode_decode(inlist):
    A = [] 
    for car in inlist :
        dec = whisper_processor.tokenizer.encode(car)
        print(f"{car} : {dec}")
        A.append(dec)

    for dec in A : 
        print(f"{dec} : {whisper_processor.tokenizer.decode(dec)}")

    for i in range(len(inlist)) : 
        print(f"{inlist[i]} : {whisper_processor.tokenizer.decode(A[i], skip_special_tokens=True)}")

    

def test_bltk():
    print(f'Vowels: {vowels}')
    display_encode_decode(vowels)
    print(f'Vowel signs: {vowel_signs}')
    display_encode_decode(vowel_signs)
    print(f'Consonants: {consonants}')
    display_encode_decode(consonants)
    print(f'Digits: {digits}')
    display_encode_decode(digits)
    print(f'Operators: {operators}')
    display_encode_decode(operators)
    print(f'Punctuation marks: {punctuations}')
    display_encode_decode(punctuations)
    print(f'Others: {others}')
    display_encode_decode(others)



test_bltk()


print("PAAAAAAad")
print(whisper_processor.tokenizer.pad_token_id)

print("bos_token_id")

print(whisper_processor.tokenizer.bos_token_id)

print("eos_token_id")
print(whisper_processor.tokenizer.eos_token_id)






