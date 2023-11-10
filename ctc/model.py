import torchaudio
import torch
import torch.nn as nn 
from common.common_params import encoder_dim, predictor_dim, joiner_dim, embedding_size, BATCH_SIZE, MFCC_N, ENCODER_TIME_DIM_INPUT_SIZE


class CTCModel(nn.Module) : 
    
    def __init__(self, vocabulary_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3),  padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3),  padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3),  padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3),  padding=(1, 1)),
            nn.ReLU(),
        )
        encoder_input_size = 256 * 13# 1280 
        self.linear = nn.Linear(encoder_input_size, vocabulary_size)

    def forward(self, x):
        batch_size, _, _, len_seq = x.shape

        out = self.cnn(x)
        out = out.permute(0,3,2, 1).reshape(batch_size, len_seq, -1)

        out = self.linear(out)
        return out