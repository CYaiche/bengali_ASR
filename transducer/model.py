# Code based on  Loren Lugosch in transducer tutorial on github 
# https://github.com/lorenlugosch/transducer-tutorial/blob/main/transducer_tutorial_example.ipynb

# Using pytorch lightning for easier data loading  

from transducer.encoder_decoder_joiner import Encoder, Predictor , Joiner
import lightning.pytorch as pl
import torch


class Transducer(pl.LightningModule):
    def __init__(self, input_size, num_outputs):
        """_summary_

        Args:
            input_size (int): number of mel frames
            num_outputs (int): _description_
        """
        super(Transducer, self).__init__()
        self.encoder    = Encoder(input_size)
        self.predictor  = Predictor(num_outputs)
        self.joiner     = Joiner(num_outputs)
        
        
        self.T          = torch.zeros(input_size)
        self.U          = torch.zeros(num_outputs)
        
        
    def compute_forward_prob(self, joiner_out, T, U, y):
        """
        joiner_out: tensor of shape (B, T_max, U_max+1, #labels)
        T: list of input lengths
        U: list of output lengths 
        y: label tensor (B, U_max+1)
        """
        B = joiner_out.shape[0]
        T_max = joiner_out.shape[1]
        U_max = joiner_out.shape[2] - 1
        log_alpha = torch.zeros(B, T_max, U_max+1).to(model.device)
        for t in range(T_max):
            for u in range(U_max+1):
                if u == 0:
                    if t == 0:
                        log_alpha[:, t, u] = 0.

                    else: #t > 0
                        log_alpha[:, t, u] = log_alpha[:, t-1, u] + joiner_out[:, t-1, 0, NULL_INDEX] 
                        
                else: #u > 0
                    if t == 0:
                        log_alpha[:, t, u] = log_alpha[:, t,u-1] + torch.gather(joiner_out[:, t, u-1], dim=1, index=y[:,u-1].view(-1,1) ).reshape(-1)
                    
                    else: #t > 0
                        log_alpha[:, t, u] = torch.logsumexp(torch.stack([
                            log_alpha[:, t-1, u] + joiner_out[:, t-1, u, NULL_INDEX],
                            log_alpha[:, t, u-1] + torch.gather(joiner_out[:, t, u-1], dim=1, index=y[:,u-1].view(-1,1) ).reshape(-1)
                        ]), dim=0)
            
        log_probs = []
        for b in range(B):
            log_prob = log_alpha[b, T[b]-1, U[b]] + joiner_out[b, T[b]-1, U[b], NULL_INDEX]
            log_probs.append(log_prob)
            log_probs = torch.stack(log_probs) 
        return log_prob
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        encoder_out     = self.encoder.forward(x)
        predictor_out   = self.predictor.forward(y)
        joiner_out      = self.joiner.forward(encoder_out.unsqueeze(2), predictor_out.unsqueeze(1)).log_softmax(3)

        loss = -self.compute_forward_prob(joiner_out, self.T, self.U, y).mean()
        
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
# sanity check   
if __name__ == "__main__" : 
    model = Transducer(1024,1024)