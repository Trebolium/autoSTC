from torch.nn import Linear, BatchNorm1d, Parameter
import torch, math, pdb
import torch.nn as nn
import torch.nn.functional as F

class Vt_Embedder(nn.Module):
    def __init__(self, config, spmel_params):
        super().__init__()

        """ Choi's k2c2 """
        # the number of chunks that a spectrogram is split into
        self.chunk_num = config.chunk_num
        self.dropout = 0
        self.batch_size = config.batch_size
        self.kernelSize = 3
        # max pooling seems to be row-column 
        self.paddingSize = int(math.ceil((self.kernelSize-1)/2))
        self.inc1Dim = 512
        self.final_embedding = 256
        self.latent_dim = 64
        self.num_classes= 6 # number of classes
        self.n_mels = spmel_params['n_mels']
        melsteps_per_second = spmel_params['sr'] / spmel_params['hop_size']
        self.window_size = math.ceil(config.chunk_seconds * melsteps_per_second)
        self.maxpool_factor_height = int(self.n_mels/2/3/4)
        self.maxpool_factor_width = int(self.window_size/2/2/2)
        self.is_blstm = True
        self.lstm_num = 2

        self.conv_layer1 = nn.Sequential(
                nn.Conv2d(1
                            ,64
                            ,kernel_size = self.kernelSize, padding = self.paddingSize)
                ,nn.BatchNorm2d(64)
                ,nn.ReLU()
                ,nn.MaxPool2d(2,2)
            )
        self.conv_layer2 = nn.Sequential(
                nn.Conv2d(64
                            ,128
                            ,kernel_size = self.kernelSize, padding = self.paddingSize)
                ,nn.BatchNorm2d(128)
                ,nn.ReLU()
                ,nn.MaxPool2d(self.maxpool_factor_width, self.maxpool_factor_height)
            )
        self.conv_layer3 = nn.Sequential(
                nn.Conv2d(128
                            ,256
                            ,kernel_size = self.kernelSize, padding = self.paddingSize)
                ,nn.BatchNorm2d(256)
                ,nn.ReLU()
                ,nn.MaxPool2d(2,3)
            )   
        self.conv_layer4 = nn.Sequential(
                nn.Conv2d(256
                            ,512
                            ,kernel_size = self.kernelSize, padding = self.paddingSize)
                ,nn.BatchNorm2d(512)
                ,nn.ReLU()
                ,nn.MaxPool2d(2,4)
            )   
        self.flat_size = 512

        """ Dense Layers """

        self.fc_layer1 = nn.Sequential(
            nn.Linear(512
                        ,512)
            ,nn.Dropout(self.dropout)
            ,nn.BatchNorm1d(512)
            ,nn.ReLU()
            )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(512
                        ,256)
            #,nn.Dropout(self.dropout)
            ,nn.BatchNorm1d(256)
            ,nn.ReLU()
            )

        if self.is_blstm == True:
            lstm_mult = 2
        else:
            lstm_mult = 1

        #fc_layer3_dim = 1536
        fc_layer3_dim = 256 * lstm_mult
        self.fc_layer3 = nn.Sequential(
            nn.Linear(fc_layer3_dim
                        ,self.latent_dim)
            ,nn.BatchNorm1d(self.latent_dim)
            ,nn.ReLU()
            )

        self.fc_layer4 = nn.Sequential(
            nn.Linear(self.latent_dim * self.chunk_num
                        ,self.final_embedding)
            ,nn.BatchNorm1d(self.final_embedding)
            ,nn.ReLU()
            )

        """ BLSMT layers """

        self.lstm = nn.LSTM(256, 256, self.lstm_num, batch_first=True, bidirectional=self.is_blstm)

        """ Attention Layer"""
        # Make a 1layer FFNN for each weight in the network
        #self.feat2weight_ffnn = nn.Linear(self.latent_dim, 1)

        """ Classification Layer """

#        self.class_layer_wAttn = nn.Sequential(
#            nn.Linear(self.latent_dim, self.num_classes)
#            )
        self.class_layer_noAttn = nn.Sequential(
                    nn.Linear(self.final_embedding, self.num_classes)
                                )

    def forward(self, x): 
#        if self.file_name == 'defaultName': pdb.set_trace()
        x0 = x.transpose(-1,-2)
        x0 = x0.unsqueeze(1) # for 2D convolution
        xc1 = self.conv_layer1(x0)
        xc2 = self.conv_layer2(xc1)
        xc3 = self.conv_layer3(xc2)
        xc4 = self.conv_layer4(xc3)
        xc4_squeezed = xc4.squeeze(3).squeeze(2) 
        xfc2 = self.fc_layer2(self.fc_layer1(xc4_squeezed))
        xfc2_by_example = xfc2.view(xfc2.shape[0] // self.chunk_num, self.chunk_num, xfc2.shape[1])
        #collect all chunks of the same example and send them in groups to the BLSTM
        self.lstm.flatten_parameters()
        # the first value returned by LSTM is all of the hidden states throughout
        # the sequence. the second is just the most recent hidden state
        lstm_outs, hidden = self.lstm(xfc2_by_example)


        #https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
        lstm_outs = lstm_outs.contiguous()
        # flattened chunks across the batch axis, so next fc layer can process all chunks at once
        lstm_outs_by_chunk = lstm_outs.view(lstm_outs.shape[0] * lstm_outs.shape[1], lstm_outs.shape[2])
        fc3_by_chunk = self.fc_layer3(lstm_outs_by_chunk)
        # group by tensor again so that attention mechanism can consider all chunks for context vector
        fc3_by_example = fc3_by_chunk.view(fc3_by_chunk.shape[0]//self.chunk_num, self.chunk_num, fc3_by_chunk.shape[1])
#        if self.file_name == 'defaultName':
#            print('x.shape', x.shape)
#            print('x0.shape', x0.shape)
#            print('xc1.shape', xc1.shape)
#            print('xc2.shape', xc2.shape)
#            print('xc3.shape', xc3.shape)
#            print('xc4.shape', xc4.shape)
#            print('xc4_squeezed.shape', xc4_squeezed.shape)
#            print('xfc2.shape', xfc2.shape)
#            print('xfc2_by_example.shape', xfc2_by_example.shape)
#            print('lstm_outs.shape', lstm_outs.shape)
#            print('lstm_outs_by_chunk.shape', lstm_outs_by_chunk.shape)
#            print('fc3_by_chunk', fc3_by_chunk.shape)
#            print('fc3_by_example.shape', fc3_by_example.shape)
#            pdb.set_trace()
        flattened_fc3_example = fc3_by_example.view(fc3_by_example.shape[0], fc3_by_example.shape[1]*fc3_by_example.shape[2])
        emb = self.fc_layer4(flattened_fc3_example)
        prediction = self.class_layer_noAttn(emb)
        all_tensors = [xc1, xc2, xc3, xc4, xfc2, lstm_outs, fc3_by_chunk, emb]
        return prediction, all_tensors

def my_loss_function(prediction, target):
    loss = F.cross_entropy(prediction, target)
    return loss
