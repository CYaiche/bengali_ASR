import os 
import json 
import pandas as pd
import torch 
from torch.utils.data import  DataLoader
from data_preparation.loader_collate import collate_fn 
from common.common_params import EXTRACT_DIR, BENG_DIR, BATCH_SIZE, metadata_folder
from common.common_params  import ENCODER_TIME_DIM_INPUT_SIZE, MAX_TEXT_OUTPUT, vocabulary_location, null_index, vocabulary_size

from data_preparation.CustomAudioDataset import CustomAudioDataset
from sklearn.model_selection import train_test_split


# from transducer.loss import transducer_loss
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from speechbrain.nnet.loss.transducer_loss import TransducerLoss
from ctc.model import CTCModel
from torch.nn.functional import ctc_loss, log_softmax
# **************** Data  ****************  # 

data_folder         = os.path.join(EXTRACT_DIR, "life")
metadata_path       = os.path.join(metadata_folder, "life_clean_enh.csv")

dataframe =  pd.read_csv(metadata_path)
train_dataframe, val_dataframe = train_test_split(dataframe, test_size=0.2,shuffle=True)
train_dataframe.reset_index(inplace=True, drop=True)
val_dataframe.reset_index(inplace=True, drop=True)

train_dataframe.sort_values(by="duration(s)",  inplace=True)
val_dataframe.sort_values(by="duration(s)", inplace=True)
print(train_dataframe.head())
train_dataset = CustomAudioDataset(data_folder,
                                vocabulary_location,
                                train_dataframe)

train_dataloader = DataLoader(train_dataset,
                              collate_fn=collate_fn,
                              batch_size=BATCH_SIZE, num_workers=4,
                        # shuffle=True
                        )
print(val_dataframe.head())

val_dataset = CustomAudioDataset(data_folder, vocabulary_location, val_dataframe)

validation_loader =  DataLoader(val_dataset,
                                collate_fn=collate_fn,
                                batch_size=BATCH_SIZE,num_workers=4,
                        # shuffle=True
                        )


def myloss(out, y, T, U) : 
    # permute batch size and sequence length 
    log_probs = out.permute(1,0,2).log_softmax(2).detach().requires_grad_()
    loss = ctc_loss(log_probs,y, T,U )
    return loss

# **************** training  ****************  # 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"training running on device {device}")
import torch.nn as nn

def train_one_epoch(model,
                    optimizer,
                    epoch_index, 
                    tb_writer
                    ) : 
    
    model = model.to(device)
    # t_loss = TransducerLoss(0)
    mse_loss = nn.MSELoss()
    running_loss = 0.
    last_loss = 0.    

    for i, batch in enumerate(train_dataloader):
        print(i)
        # Every data instance is an input + label pair
        x, y, T, U = batch

        x = x.to(device)
        y = y.to(device)
        T = T.to(device)
        U = U.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(x)
        print(outputs.shape)
        # Compute the loss and its gradients
        # loss = t_loss(outputs, y, T, U)
        loss = myloss(outputs, y, T, U)
        print(loss)

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)

    return model, last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.



# **************** Model  ****************  # 

model = CTCModel(vocabulary_size)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# transducer_loss = TransducerLoss(0)

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()
    model, avg_loss = train_one_epoch(model,
                                      optimizer,
                               epoch_number,
                               writer,
                            )

    running_vloss = 0.0
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        
        for i, vbatch in enumerate(validation_loader):
            x, y, T, U =  vbatch
            model = model.to(x.device)
            outputs = model(x)
            vloss = myloss(outputs, y, T, U)
            running_vloss += vloss 

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation : 
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1