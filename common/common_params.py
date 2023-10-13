import os 

REPO_DIR = "C:\dev\speech_to_text"

STT_DIR         = os.path.join(REPO_DIR, "asr_for_bengali")
EXTRACT_DIR     = os.path.join(REPO_DIR,"extracted")
SAVE_MODEL_DIR  = os.path.join(REPO_DIR,"model")


FS = 16000
BATCH_SIZE = 8 