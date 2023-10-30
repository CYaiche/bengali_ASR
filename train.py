import os
from transducer.model import Transducer
import torchaudio 
import torch 
import pandas as pd 
import json
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from common.common_params import STT_DIR, ENCODER_TIME_DIM_INPUT_SIZE, MAX_AUDIO_INPUT_LENGTH_IN_S, EXTRACT_DIR, SAVE_MODEL_DIR, FS, BATCH_SIZE, MAX_TEXT_OUTPUT, metadata_folder
from data_preparation.dataloader import CustomAudioDataset
from data_preparation.RNNTDataModule import RNNTDataModule
from lightning.pytorch.callbacks import ModelSummary


if __name__ == "__main__" : 
    # in path 
    data_folder         = os.path.join(EXTRACT_DIR,"life")
    
    text_metadata       = os.path.join(metadata_folder, "life_clean.csv")

    vocabulary_location = os.path.join(STT_DIR,"embedding","vocab_indicwav2vec.json")
    with open(vocabulary_location) as vocabulary_file:
        char_voc = vocabulary_file.read()
    vocabulary_to_id = json.loads(char_voc) 
    null_index = vocabulary_to_id["<pad>"]
    # out path
    save_model_path     = os.path.join(SAVE_MODEL_DIR,"transducer.pt")
    metadata            = pd.read_csv(text_metadata)
    print(metadata.shape)
    nb_pair = metadata.shape[0]
    max_length_sec = MAX_AUDIO_INPUT_LENGTH_IN_S
    max_label_size = MAX_TEXT_OUTPUT 
    
    # dataset = CustomAudioDataset(data_folder, vocabulary_location, metadata, FS, max_label_size, max_input_length=max_length_sec)
    # train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    datamodule = RNNTDataModule(data_folder,
                                vocabulary_location,
                                text_metadata, 
                                max_label_size,
                                max_input_length=max_length_sec)
    # most of papers ponder upon extended output space (vocabulary + null index)
    # here I choose to include tokens like BOS, EOS and UNK for later improvements
    # NULL_INDEX or NULL_TOKEN for RNN-T is equivalent of <pad> with index 0 of the vocabulary in my algorithm 
    # and already included in the vocabulary size
    vocabulary_size = len(vocabulary_to_id) 
    print(f"vocabulary_size : {vocabulary_size}")
    input_size = ENCODER_TIME_DIM_INPUT_SIZE
    print(f"encoder_input_size : {input_size}")
    
    model = Transducer(input_size, max_label_size, vocabulary_size, null_index=null_index) # +1 for start char 

    trainer = pl.Trainer(max_epochs=1,
                        # track_grad_norm=2,
                        callbacks=[ModelSummary(max_depth=1)],
                        # weights_summary='full'
                        )
    
    trainer.fit(model, datamodule)
    
    torch.save(model.state_dict(), save_model_path)
    
        
