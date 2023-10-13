# Code based on  Loren Lugosch in transducer tutorial on github
# The transducer originally for text_to_text conversion, here is adapted for audio speech as input 
# https://github.com/lorenlugosch/transducer-tutorial/blob/main/transducer_tutorial_example.ipynb

from transducer.params import NULL_INDEX, encoder_dim, predictor_dim, joiner_dim, n_mel
import torchaudio
import torch
import torch.nn as nn 


class Encoder(nn.Module):
    def __init__(self, input_sequence_size):
        super(Encoder, self).__init__()
        
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
        self.rnn = nn.GRU(input_size=1280, hidden_size=encoder_dim, num_layers=3, batch_first=True, bidirectional=True, dropout=0.1)
        self.linear = nn.Linear(encoder_dim*2, joiner_dim)
    
    
    def forward(self, x):
        out = x
        # (batch_size, depth, frequency , time )
        out = self.cnn(out)
        len_seq = out.shape[3]
        out = out.permute(0,3 ,2, 1).reshape(8, len_seq, -1)
        # (batch_size, time, depth, frequency  )
        out = self.rnn(out)[0]
        out = self.linear(out)
        return out

class Predictor(nn.Module):
    def __init__(self, num_outputs):
        super(Predictor, self).__init__()
        # self.embed = torch.nn.Embedding(num_outputs, predictor_dim)
        self.rnn = nn.GRUCell(input_size=predictor_dim, hidden_size=predictor_dim)
        self.linear = nn.Linear(predictor_dim, joiner_dim)
        
        self.initial_state = nn.Parameter(torch.randn(predictor_dim))
        self.start_symbol = NULL_INDEX # In the original paper, a vector of 0s is used; just using the null index instead is easier when using an Embedding layer.

    def forward_one_step(self, input, previous_state):
        embedding = self.embed(input)
        state = self.rnn.forward(embedding, previous_state)
        out = self.linear(state)
        return out, state

    def forward(self, y):
        batch_size = y.shape[0]
        U = y.shape[1]
        outs = []
        state = torch.stack([self.initial_state] * batch_size).to(y.device)
        for u in range(U+1): # need U+1 to get null output for final timestep 
            if u == 0:
                decoder_input = torch.tensor([self.start_symbol] * batch_size).to(y.device)
            else:
                decoder_input = y[:,u-1]
        out, state = self.forward_one_step(decoder_input, state)
        outs.append(out)
        out = torch.stack(outs, dim=1)
        return out
    
class Joiner(torch.nn.Module):
    def __init__(self, num_outputs):
        super(Joiner, self).__init__()
        self.linear = torch.nn.Linear(joiner_dim, num_outputs)

    def forward(self, encoder_out, predictor_out):
        out = encoder_out + predictor_out
        out = torch.nn.functional.relu(out)
        out = self.linear(out)
        return out
    
# sanity check   
if __name__ == "__main__" : 

    model = Encoder()
    model = Predictor(1024)
    model = Joiner(1024)