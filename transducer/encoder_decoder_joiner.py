# Code based on  Loren Lugosch in transducer tutorial on github
# The transducer originally for text_to_text conversion, here is adapted for audio speech as input 
# https://github.com/lorenlugosch/transducer-tutorial/blob/main/transducer_tutorial_example.ipynb

import torchaudio
import torch
import torch.nn as nn 
from common.common_params import encoder_dim, predictor_dim, joiner_dim, embedding_size, BATCH_SIZE, MFCC_N, ENCODER_TIME_DIM_INPUT_SIZE

class Encoder(nn.Module):
    def __init__(self, cnn_at_input=False):
        super().__init__()
        
        self.cnn_at_input = cnn_at_input
        
        if cnn_at_input : 
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(),
            )
        
            # GRU input_size : by default (L,N,H), L sequence length, N batch size and (N,L,H) when batch_first
            # 256 * 5 (leftover of 80 after conv)
            encoder_input_size = 256 *5 # 1280 
        else : 
            encoder_input_size = MFCC_N
        
        self.rnn = nn.GRU(input_size=encoder_input_size, hidden_size=encoder_dim, num_layers=3, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(encoder_dim, joiner_dim)
    

    def forward(self, x):
        # out = x
        # batch_size = x.shape[0]
        # len_seq =  x.shape[3]

        batch_size, _, _, len_seq = x.shape

        # (batch_size, depth, frequency , time )
        if self.cnn_at_input : 
            out = self.cnn(out)
            len_seq = out.shape[3]
            out = out.permute(0,3 ,2, 1).reshape(batch_size, len_seq, -1)
            # (batch_size, time, depth*frequency  )
        else : 
            len_seq = x.shape[3]
            # (batch_size, time, frequency)  
            x = x.permute(0,3 ,2, 1).reshape(batch_size, len_seq, -1)

        x, _ = self.rnn(x)
        x = self.linear(x)

        return x

class Predictor(nn.Module):
    def __init__(self, max_transcript_length, vocabulary_size, token_length=1):
        super().__init__()
        # ONE-hot encoding input
        self.emb    = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=predictor_dim)
        self.rnn    = nn.GRU(input_size=predictor_dim, hidden_size=predictor_dim, num_layers=1, batch_first=True, dropout=0.0)
        # self.rnn    = nn.GRUCell(input_size=predictor_dim, hidden_size=predictor_dim)
        self.linear = nn.Linear(predictor_dim, joiner_dim)
        
        # self.initial_state = nn.Parameter(torch.randint(0, max_transcript_length, (predictor_dim,), dtype=torch.float))

    # def forward_one_step(self, input, previous_state):
    #     embedding   = self.emb(input)
    #     state       = self.rnn.forward(embedding, previous_state)
    #     out         = self.linear(state)
    #     return out, state

    def forward(self, x):
        # print(f"y {y}")
        # print(f"y.shape {y.shape}")

        emb = self.emb(x)
        out, _ = self.rnn(emb)
        out = self.linear(out)

        # batch_size = y.shape[0]
        # U = y.shape[1] # text  max length in character
        # outs = []
        # state = torch.stack([self.initial_state] * batch_size)
        # print('y:', y.shape)


        # for u in range(U): # loop over each character
        #     # batch_idx_u = 
        #     # decoder_input = torch.tensor(y[:,u], dtype=torch.float)[:, None]
        #     # print(decoder_input.shape)
        #     out, state = self.forward_one_step(y[:,u], state)
        #     outs.append(out)
        # out = torch.stack(outs, dim=1)
        return out
    
class Joiner(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.linear = nn.Linear(joiner_dim, num_outputs)

    def forward(self, encoder_out, predictor_out):

        out = encoder_out.unsqueeze(2) + predictor_out.unsqueeze(1)
        # out = torch.nn.functional.relu(out)
        out = self.linear(out)
        # out =  out.log_softmax(dim=3)
        
        return out 
    
# sanity check   
if __name__ == "__main__" : 

    model = Encoder()
    model = Predictor(1024)
    model = Joiner(1024)