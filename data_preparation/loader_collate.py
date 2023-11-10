import torch 
import torchaudio
from common.common_params import FS, MFCC_N, MFCC_HOP_LENGTH, MFCC_N_FFT, N_MELS


def collate_fn(batch):
    batch_size = len(batch)
    T = [ pcm16k.shape[1]  for (pcm16k,_) in batch]
    U = [ len(sentence_tokenized) -1 for (_,sentence_tokenized) in batch]
    max_t = max(T)
    max_u = max(U)
    spectrograms = []
    sentence_tokenized_pad_s =[]

    mfcc   =  torchaudio.transforms.MFCC(sample_rate=FS,       
                    n_mfcc=MFCC_N,    # Number of MFCC coefficients
                    # 25ms FFT window  10ms frame shift
                    melkwargs={'n_fft': MFCC_N_FFT,
                                "n_mels": N_MELS,
                               'hop_length': MFCC_HOP_LENGTH}  
            )

    for index in range(batch_size) : 
        pcm16k, sentence_tokenized = batch[index]
        # pad audio and mffcc
        pad_right = int( max_t - pcm16k.shape[1])
        assert pad_right >=0 , "error , one audio file superior to max_input_length 16k"
        pcm16kpad = torch.nn.functional.pad(pcm16k, (0, pad_right) , "constant", 0)
        spectrogram = mfcc(pcm16kpad)
        # pad tokens 
        pad_right = int(max_u  - len(sentence_tokenized))
        sentence_tokenized_pad = torch.nn.functional.pad(torch.tensor(sentence_tokenized), (0, pad_right) , "constant", 0)

        spectrograms.append(spectrogram)
        sentence_tokenized_pad_s.append(sentence_tokenized_pad)
        
    spectrograms = torch.stack(spectrograms)
    sentence_tokenized_pad_s = torch.stack(sentence_tokenized_pad_s)
    T_f = [int(t / MFCC_HOP_LENGTH) +1  for t in T] # becuse of mfcc padding
    return (spectrograms, sentence_tokenized_pad_s, torch.tensor(T_f),  torch.tensor(U)) 



