
import torch 
import torch.nn as nn
from common.common_params import USE_SPEECHBRAIN_LOSS
if USE_SPEECHBRAIN_LOSS : 
    from speechbrain.nnet.loss.transducer_loss import TransducerLoss

from transducer.encoder_decoder_joiner import Encoder, Predictor , Joiner


class TransducerModel(nn.Module):

    def __init__(self, input_size, max_transcript_length, vocabulary_size, null_index=0):
        super().__init__()

        self.encoder    = Encoder()
        self.predictor  = Predictor(max_transcript_length, vocabulary_size)
        self.joiner     = Joiner(vocabulary_size)

        # dummy network for test : 
        self.dummy_network = False
        self.dummy_linear = nn.Linear(100, vocabulary_size)
        # if USE_SPEECHBRAIN_LOSS : 
        #     self.transducer_loss = TransducerLoss(0)
        
        # self.T          = torch.zeros(input_size)
        # self.U          = torch.zeros(max_transcript_length)
        # self.null_index = null_index

    def forward(self, x, y) : 
        if self.dummy_network  : 
            # dummy 
            out =  self.encoder(x)
            joiner_out = self.dummy_linear(x)
            return out 
        encoder_out     = self.encoder(x)
        predictor_out   = self.predictor(y)
        joiner_out      = self.joiner(encoder_out, predictor_out)

        print(joiner_out.shape)
        return joiner_out

