import os 
# os

if os.name =='nt':
    GPU_ON_SYSTEM = False
else : 
    GPU_ON_SYSTEM = True
    
REPO_DIR = "C:\dev\speech_to_text"

STT_DIR         = os.path.join(REPO_DIR, "asr_for_bengali")
EXTRACT_DIR     = os.path.join(REPO_DIR,"extracted")
SAVE_MODEL_DIR  = os.path.join(REPO_DIR,"model")
metadata_folder = os.path.join(EXTRACT_DIR,"metadata")


MAX_AUDIO_INPUT_LENGTH_IN_S = 10.63
MAX_TEXT_OUTPUT  = 128
MAX_TEXT_OUTPUT_MINUS_START_CHAR = MAX_TEXT_OUTPUT - 1 
FS = 16000
BATCH_SIZE = 4
START_TOKEN = "<s>" # BOS
END_TOKEN  = "</s>" # EOS
PAD_TOKEN = "<pad>" # PAD equivalent to null index



# MFCC
MFCC_N_MELS = 80 
MFCC_N_FFT = 400
MFCC_HOP_LENGTH = 160
ENCODER_TIME_DIM_INPUT_SIZE = int((FS*MAX_AUDIO_INPUT_LENGTH_IN_S )/ MFCC_HOP_LENGTH)

# RNN : encoder and joiner dim must be > to vocab 
encoder_dim     = 100 # 1024
predictor_dim   = 40
joiner_dim      = 100 # 1024
embedding_size  = 100 
