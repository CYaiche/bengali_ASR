# Code based on  Loren Lugosch in transducer tutorial on github 
# https://github.com/lorenlugosch/transducer-tutorial/blob/main/transducer_tutorial_example.ipynb

# Using pytorch lightning for easier data loading  
import os 
import lightning.pytorch as pl
import torch
from common.common_params import USE_SPEECHBRAIN_LOSS
if USE_SPEECHBRAIN_LOSS : 
    from speechbrain.nnet.loss.transducer_loss import TransducerLoss

from transducer.encoder_decoder_joiner import Encoder, Predictor , Joiner

class LightningTransducer(pl.LightningModule):
    def __init__(self, input_size, max_transcript_length, vocabulary_size, null_index=0):
        """_summary_

        Args:
            input_size (int): number of mel frames
            num_outputs (int): _description_
        """
        super(LightningTransducer, self).__init__()
        print("Transducer init")
        self.encoder    = Encoder()
        self.predictor  = Predictor(max_transcript_length, vocabulary_size)
        self.joiner     = Joiner(vocabulary_size)
        if USE_SPEECHBRAIN_LOSS : 
            self.transducer_loss = TransducerLoss(0)
        
        self.T          = torch.zeros(input_size)
        self.U          = torch.zeros(max_transcript_length)
        self.null_index = null_index

        # self.automatic_optimization = False # use manual retropropagation

    def compute_forward_prob(self, joiner_out,T, U, y):
        """
        joiner_out: tensor of shape (B, T_max, U_max, #labels)
        T: list of input lengths
        U: list of output lengths 
        y: label tensor (B, U_max)
        """
        null_idx = self.null_index
        batch_size = joiner_out.shape[0]
        T_max = joiner_out.shape[1] # new time axis length , is it ok ?? 
        U_max = joiner_out.shape[2]
        log_alpha = torch.zeros(batch_size, T_max, U_max).to("cuda:0") # change this if remove pytorch lignthing 

        # base on forward-backward algorithm see "sequence Transduction with RNN by Alex Graves"
        # all computation are "+" here, are we are in log space, equivalent to "*" in the paper
        print(f"T_max : {T_max}")
        print(f"U_max : {U_max}")
        print("compute_forward_prob : begin ")
        for t in range(T_max):
            for u in range(U_max):
                if u == 0: 
                    if t == 0:
                        # here t begins at 1 different that in the paper 
                        # and probability of having a <BOS> is 1 leads to 0 as log(1) = 0 
                        log_alpha[:, t, u] = 0. 

                    else: #t > 0 # did not write a label yet
                        log_alpha[:, t, u] = log_alpha[:, t-1, u] + joiner_out[:, t-1, u, null_idx] 
                        
                else: # u > 0
                    # probability of only outputting u labels 
                    if t == 0:
                        log_alpha[:, t, u] = log_alpha[:, t, u-1] + torch.gather(joiner_out[:, t, u-1], dim=1, index=y[:,u-1].view(-1,1) ).reshape(-1)
                    # probability of outputting t null index and u labels 
                    else: #t > 0
                        log_alpha[:, t, u] = torch.logsumexp(torch.stack([
                            log_alpha[:, t-1, u] + joiner_out[:, t-1, u, null_idx],
                            log_alpha[:, t, u-1] + torch.gather(joiner_out[:, t, u-1], dim=1, index=y[:,u-1].view(-1,1) ).reshape(-1)
                        ]), dim=0)
        print("compute_forward_prob : end log_alpha ")
        log_probs = []
        for b in range(batch_size):
            log_prob = log_alpha[b, T[b]-1, U[b]-1] + joiner_out[b, T[b]-1, U[b]-1, null_idx]
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs) 
        print(log_probs)
        print("compute_forward_prob : end log_probs ")

        return log_probs
    
    def training_step(self, batch, batch_idx):
        print("begin training step ")
        x, y, T, U  = batch
        
        encoder_out     = self.encoder.forward(x)
        predictor_out   = self.predictor.forward(y)
        joiner_out      = self.joiner.forward(encoder_out, predictor_out)

        if USE_SPEECHBRAIN_LOSS : 
            print("transducer_loss : begin  ")
            loss = self.transducer_loss(joiner_out, y, T, U)
            print("transducer_loss : end begin ")
        else : 
            loss = -self.compute_forward_prob(joiner_out, T, U, y).mean()

        # self.manual_backward(loss)
        # # loss.backward()
        # opt = self.optimizers()
        # opt.zero_grad()
        # opt.step()

        tensorboard = self.logger.experiment
        tensorboard.add_scalars("loss", {"train": loss}, self.global_step)
        print("end training step ")
        print(loss)
        return  loss

    def validation_step(self, batch, batch_idx):
        print("begin validation_step ")
        x, y, T, U  = batch
        
        encoder_out     = self.encoder.forward(x)
        predictor_out   = self.predictor.forward(y)
        joiner_out      = self.joiner.forward(encoder_out, predictor_out)

        if USE_SPEECHBRAIN_LOSS : 
            loss = self.transducer_loss(joiner_out, y, T, U)
        else : 
            loss = -self.compute_forward_prob(joiner_out, T, U, y).mean()
        
        tensorboard = self.logger.experiment
        tensorboard.add_scalars("loss", {"val": loss}, self.global_step)
        print("end validation_step ")
        return  loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# sanity check   
if __name__ == "__main__" : 
    model = LightningTransducer(1024,1024)