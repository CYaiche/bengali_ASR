import os
import torch 


if os.name =='nt':
    USE_SPEECHBRAIN_LOSS = False
elif os.name =='posix' : 
    USE_SPEECHBRAIN_LOSS = True # for now to test
else : 
    USE_SPEECHBRAIN_LOSS = False
    

STT_DIR         = "/home/nxp66145/clara/"
BENG_DIR        = os.path.join(STT_DIR, "bengali_ASR")
EXTRACT_DIR     = os.path.join(STT_DIR, "..", "extracted")
if os.name =='posix' : 
    # EXTRACT_DIR = os.path.join(STT_DIR, "extracted")
    EXTRACT_DIR = os.path.join("/disk2", "clara","extracted")
SAVE_MODEL_DIR  = os.path.join(STT_DIR, "..", "model")
metadata_folder = os.path.join(EXTRACT_DIR, "metadata")


MAX_AUDIO_INPUT_LENGTH_IN_S = 10.63
MAX_TEXT_OUTPUT  = 128
MAX_TEXT_OUTPUT_MINUS_START_CHAR = MAX_TEXT_OUTPUT - 1 
FS = 16000
BATCH_SIZE  = 32
START_TOKEN = "<s>" # BOS
END_TOKEN   = "</s>" # EOS
PAD_TOKEN   = "<pad>" # PAD equivalent to null index



# MFCC
MFCC_N   = 13 # 80 
MFCC_N_FFT      = 512
MFCC_HOP_LENGTH = 256
N_MELS = 80 
ENCODER_TIME_DIM_INPUT_SIZE = int((FS*MAX_AUDIO_INPUT_LENGTH_IN_S )/ MFCC_HOP_LENGTH)

# RNN : encoder and joiner dim must be > to vocab 
encoder_dim     = 100 # 1024
predictor_dim   = 40
joiner_dim      = 100 # 1024
embedding_size  = 100 


# vocabulary 
vocabulary_location = os.path.join(BENG_DIR,"embedding","vocab_indicwav2vec.json")
vocabulary_size     =  torch.load(os.path.join(BENG_DIR, "embedding",'vocabulary_size.pt'))
null_index          = 0  # vocabulary_to_id["<pad>"]
