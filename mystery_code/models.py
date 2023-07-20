from torch.nn import Linear, BatchNorm1d, Parameter
import torch, math, pdb
import torch.nn as nn
import torch.nn.functional as F

#remove this comment
class Luo2019AsIs(nn.Module):
    def __init__(self, config, spmel_params):
        super().__init__()
        """ 1DKernel 1D CNN layers"""
        # the number of chunks that a spectrogram is split into
        self.use_attention = config.use_attention
        self.chunk_num = config.chunk_num
        self.batch_size = config.batch_size
        self.dropout = config.dropout
        self.kernelSize = 3
        self.paddingSize = int(math.ceil((self.kernelSize-1)/2))
        self.initial_channels = config.n_mels
        self.inc1Dim = 512
        self.latent_dim = 16
        self.num_classes=6 # number of classes
        melsteps_per_second = spmel_params['sr'] / spmel_params['hop_size']
        self.window_size = math.ceil(config.chunk_seconds * melsteps_per_second)
        self.is_blstm = config.is_blstm
        self.lstm_num = config.lstm_num
        self.file_name = config.file_name

        self.conv_layer1 = nn.Sequential(
                nn.Conv1d(96
                            ,512
                            ,kernel_size = 3, padding = self.paddingSize)
                ,nn.BatchNorm1d(512)
                ,nn.ReLU()
            )
        self.conv_layer2 = nn.Sequential(
                nn.Conv1d(512
                            ,512
                            ,kernel_size = 3, padding = self.paddingSize)
                ,nn.BatchNorm1d(512)
                ,nn.ReLU()
            )

        self.flat_size = 44 * 512

        """ Dense Layers """

        self.fc_layer1 = nn.Sequential(
            nn.Linear(self.flat_size
                        ,512)
            ,nn.Dropout(self.dropout)
            ,nn.BatchNorm1d(512)
            ,nn.ReLU()
            )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(512
                        ,256)
            ,nn.BatchNorm1d(256)
            ,nn.ReLU()
            )

        if self.is_blstm == True:
            lstm_mult = 2
        else:
            lstm_mult = 1

#        fc3_layers = []
#        for i in range(self.chunk_num):
#            fc3_layer = nn.Sequential(
#                nn.Linear(fc_layer3_dim
#                    ,self.latent_dim)
#                ,nn.BatchNorm1d(self.latent_dim)
#                ,nn.ReLU()
#                )
#            fc3_layers.append(fc3_layer)
#        self.fc_by_chunk = nn.ModuleList(fc3_layers)

        fc_layer3_dim = 256 * lstm_mult
        self.fc_layer3 = nn.Sequential(
            nn.Linear(fc_layer3_dim
                        ,self.latent_dim)
            ,nn.BatchNorm1d(16)
            ,nn.ReLU()
            )

        """ BLSMT layers """

        self.lstm = nn.LSTM(256, 256, self.lstm_num, batch_first=True, bidirectional=self.is_blstm)

        """ Attention Layer"""
##############################################################################
        # Make a 1layer FFNN for each weight in the network

        self.feat2weight_ffnn = nn.Linear(self.latent_dim, 1)
#        feature_to_weight_functions = []
#        for i in range(self.chunk_num):
#            # for each value of h_j, create a corresponding weight
#            hiddenV2weightV_layer = nn.Linear(self.latent_dim, self.latent_dim)
#            feature_to_weight_functions.append(hiddenV2weightV_layer)
#        self.f2w_functions = nn.ModuleList(feature_to_weight_functions)

##############################################################################

        """ Classification Layer """

        self.class_layer_wAttn = nn.Sequential(
            nn.Linear(self.latent_dim, self.num_classes)
            )
        
        self.class_layer_noAttn = nn.Sequential(
            nn.Linear(self.latent_dim * self.chunk_num, self.num_classes)
            )

    def forward(self, x):
        if self.file_name == 'defaultName': pdb.set_trace()
        x = x.transpose(-1,-2)
        xc1 = self.conv_layer1(x)
        xc2 = self.conv_layer2(xc1)
        # num_examples not always batch size, if Dataloader's drop_last=true
        num_examples = int(xc2.shape[0]/self.chunk_num)
    	# flatten for next fc layers
        flattened_xc2 = xc2.view(xc2.shape[0], -1)
        xfc2 = self.fc_layer2(self.fc_layer1(flattened_xc2))
        # group tensor by chunks so that lstm can process all relevant timesteps of vectors
        xfc2_by_full_example = xfc2.view(xfc2.shape[0] // self.chunk_num, self.chunk_num, xfc2.shape[1])
        self.lstm.flatten_parameters()
        lstm_outs, hidden = self.lstm(xfc2_by_full_example)
        # https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
        lstm_outs = lstm_outs.contiguous()
        # flattened chunks across the batch axis, so next fc layer can process all flattened tensors simultaneously
        lstm_outs_by_batch = lstm_outs.view(lstm_outs.shape[0] * lstm_outs.shape[1], lstm_outs.shape[2])
        fc3_by_batch = self.fc_layer3(lstm_outs_by_batch)
        # group by tensor again so that attention mechanism can consider all chunks for context vector
        fc3_by_full_example = fc3_by_batch.view(fc3_by_batch.shape[0]//self.chunk_num, self.chunk_num, fc3_by_batch.shape[1])

        if self.file_name == 'defaultName': pdb.set_trace()
        # at this point the tensor is no longer temporal, so why is attention used?

        if self.use_attention == True:
            """Simplified Attention mechanism - http://arxiv.org/abs/1512.08756"""
            ###########################################################################
            for i in range(num_examples):
                weight = self.feat2weight_ffnn(fc3_by_full_example[i])
                if i == 0:
                    weights_values = weight.unsqueeze(0)
                else:
                    weights_values = torch.cat((weights_values, weight.unsqueeze(0)))

            if self.file_name == 'defaultName': pdb.set_trace()
            weights_values = weights_values.squeeze(2)
            attn_weights = F.softmax(weights_values, dim=1)
            # these weights are then applied to the 'hidden features' (multiplied together)
            attn_applied = attn_weights.unsqueeze(-1) * fc3_by_full_example
            # get the sum of all weighted features to produce context c
            context = torch.sum(attn_applied, dim=1)
            prediction = self.class_layer_wAttn(context)
            ##########################################################################

        else:
            #  flatten for proceeding classification fc layer
            flattened_fc3_full_example = fc3_by_full_example.view(fc3_by_full_example.shape[0], fc3_by_full_example.shape[1]*fc3_by_full_example.shape[2]) 
            prediction = self.class_layer_noAttn(flattened_fc3_full_example)
        return prediction


class Choi_k2c2(nn.Module):
    def __init__(self, config, spmel_params):
        super().__init__()

        """ Choi's k2c2 """
        # the number of chunks that a spectrogram is split into
        self.use_attention = config.use_attention
        self.chunk_num = config.chunk_num
        self.dropout = config.dropout
        self.batch_size = config.batch_size
        self.kernelSize = 3
        self.paddingSize = int(math.ceil((self.kernelSize-1)/2))
        self.initial_channels = config.n_mels
        self.inc1Dim = 512
        self.latent_dim = 16
        self.num_classes=6 # number of classes
        melsteps_per_second = spmel_params['sr'] / spmel_params['hop_size']
        self.window_size = math.ceil(config.chunk_seconds * melsteps_per_second)
        self.is_blstm = config.is_blstm
        self.lstm_num = config.lstm_num
        self.file_name = config.file_name

        self.conv_layer1 = nn.Sequential(
                nn.Conv2d(1,
                           64,
                            kernel_size = self.kernelSize, padding = self.paddingSize),
                nn.BatchNorm2d(64),
                nn.ReLU()
                ,nn.MaxPool2d(2,2)
            )
        self.conv_layer2 = nn.Sequential(
                nn.Conv2d(64,
                            128,
                            kernel_size = self.kernelSize, padding = self.paddingSize),
                nn.BatchNorm2d(128),
                nn.ReLU()
                ,nn.MaxPool2d(5,4)
            )
        self.conv_layer3 = nn.Sequential(
                nn.Conv2d(128,
                            256,
                            kernel_size = self.kernelSize, padding = self.paddingSize),
                nn.BatchNorm2d(256),
                nn.ReLU()
                ,nn.MaxPool2d(2,3)
            )   
        self.conv_layer4 = nn.Sequential(
                nn.Conv2d(256,
                            512,
                            kernel_size = self.kernelSize, padding = self.paddingSize),
                nn.BatchNorm2d(512),
                nn.ReLU()
                ,nn.MaxPool2d(2,4)
            )   
        self.flat_size = 512

        """ Dense Layers """

        self.fc_layer1 = nn.Sequential(
            nn.Linear(self.flat_size
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
        fc_layer3_dim = 256 * lstm_mult * self.chunk_num
        self.fc_layer3 = nn.Sequential(
            nn.Linear(fc_layer3_dim
                        ,self.latent_dim)
#            ,nn.BatchNorm1d(16)
            ,nn.ReLU()
            )

        """ BLSMT layers """

        self.lstm = nn.LSTM(256, 256, self.lstm_num, batch_first=True, bidirectional=self.is_blstm)

        """ Attention Layer"""
##############################################################################
        # Make a 1layer FFNN for each weight in the network
        
        feature_to_weight_functions = []
        for i in range(self.latent_dim):
            # for each value of h_j, create a corresponding weight
            hidden2weight_layer = nn.Linear(self.batch_size, self.batch_size)
            feature_to_weight_functions.append(hidden2weight_layer)
        self.f2w_functions = nn.ModuleList(feature_to_weight_functions)

##############################################################################

        """ Classification Layer """

        self.classify_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.num_classes)
#            ,nn.BatchNorm1d(self.num_classes)
            #,nn.Sigmoid()
            )

    def forward(self, x): 
        x0 = x.transpose(-1,-2)
        x0 = x0.unsqueeze(1) # for 2D convolution
        xc1 = self.conv_layer1(x0)
        xc2 = self.conv_layer2(xc1)
        xc3 = self.conv_layer3(xc2)
        xc4 = self.conv_layer4(xc3)
        if self.file_name == 'defaultName':
            print(x.shape)
            print(x0.shape)
            print(xc1.shape)
            print(xc2.shape)
            print(xc3.shape)
            print(xc4.shape)
            pdb.set_trace()
        
        # group tensors by their corresponding example/ audio recording
        grouped_by_recording = []
        for i in range(self.batch_size):
            offset = i * self.chunk_num
            example_batch = xc4[offset : offset + self.chunk_num]
            grouped_by_recording.append(example_batch)

        if self.file_name == 'defaultName':
            pdb.set_trace()
        
        # this block produces separately calculated dense layers that are concatenated together at the end
        for i, recording in enumerate(grouped_by_recording):
            if recording.shape[0] == self.chunk_num:
                flattened_rec = recording.squeeze(-1).squeeze(-1)
                xfc1 = self.fc_layer1(flattened_rec)
                xfc2 = self.fc_layer2(xfc1)
                if i == 0:
                    dense_by_recording = xfc2.unsqueeze(0)
                else:
                    dense_by_recording = torch.cat((dense_by_recording, xfc2.unsqueeze(0)))
        
        if self.file_name == 'defaultName':
            pdb.set_trace()

        #collect all chunks of the same example and send them in groups to the BLSTM
        self.lstm.flatten_parameters()
    	# the first value returned by LSTM is all of the hidden states throughout
	    # the sequence. the second is just the most recent hidden state
        lstm_outs, hidden = self.lstm(dense_by_recording)


        #https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
        lstm_outs = lstm_outs.contiguous()
        flattened_lstm_outs = lstm_outs.view(lstm_outs.size(0), -1)
        xfc3 = self.fc_layer3(flattened_lstm_outs)
        # at this point the tensor is no longer temporal, so why is attention used?
        """Working on Attention Layer bit"""
######################################################################################
        # this 1layer FFNN takes the hidden state produced from the lstm layer
        # and learns a function to convert it into the ideal weights
        if self.use_attention == True:
            for i in range(self.latent_dim):
                # use linear layer as f function for determining weight values from h_{j}
                weight = self.f2w_functions[i](xfc3[:,i])
                if i == 0:
                    weights_values = weight.unsqueeze(0)
                else:
                    weights_values = torch.cat((weights_values, weight.unsqueeze(0)))
            # after recombining all tensors the must be transposed for correct tensor shape (batch, features)
            weights_values = weights_values.transpose(0,1)
            # which are then soft-maxed
            attn_weights = F.softmax(weights_values, dim=1)
            # these weights are then applied to the 'hidden features' (multiplied together)
            attn_applied = attn_weights * xfc3
            # get the sum of all weighted features to produce context c
            context = torch.sum(attn_applied, dim=1)
            prediction = context
######################################################################################
        else:
            prediction = self.classify_layer(xfc3)
        return prediction


class WilkinsAudioCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.initial_channels = 1
        self.conv1_filters = 16
        self.conv2_filters = 8
        self.conv3_filters = 32
        self.max_stride = 8
        self.window_size = 3 * 44100
        self.class_num=10
        self.dropout = config.dropout

        self.layer_seq1 = nn.Sequential(
            nn.Conv1d(in_channels=self.initial_channels,
                out_channels=self.conv1_filters,
                kernel_size=128, stride=1, padding=math.ceil((128-1)/2)),
            nn.ReLU(),
            nn.BatchNorm1d(self.conv1_filters),
            nn.MaxPool1d(kernel_size=64, stride=self.max_stride, padding = math.ceil((64-1)/2))
            )

        self.layer_seq2 = nn.Sequential(
            nn.Conv1d(in_channels=self.conv1_filters,
                out_channels=self.conv2_filters,
                kernel_size=64, stride=1, padding=math.ceil((64-1)/2)),
            nn.ReLU(),
            nn.BatchNorm1d(self.conv2_filters),
            nn.MaxPool1d(kernel_size=64, stride=self.max_stride, padding = math.ceil((64-1)/2))
            )

        self.layer_seq3 = nn.Sequential(
            nn.Conv1d(in_channels=self.conv2_filters,
                out_channels=self.conv3_filters,
                kernel_size=256, stride=1, padding=math.ceil((256-1)/2)),
            nn.ReLU(),
            nn.BatchNorm1d(self.conv3_filters),
            nn.MaxPool1d(kernel_size=64, stride=self.max_stride, padding = math.ceil((64-1)/2))
            )

        self.flat_size = math.ceil(self.window_size / (self.max_stride**3)) * self.conv3_filters

        self.fc_layer1 = nn.Sequential(
            nn.Linear(self.flat_size, self.conv3_filters),
            nn.Dropout(self.dropout),
            nn.ReLU()
            )

        self.fc_layer2 = nn.Sequential(
            nn.Linear(self.conv3_filters, self.class_num),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = x.unsqueeze(1)
        out_seq1_x = self.layer_seq1(x)
        out_seq2_x = self.layer_seq2(out_seq1_x)
        out_seq3_x = self.layer_seq3(out_seq2_x)
        # retain batch_size dimension, flatten rest
        flattened_x = out_seq3_x.view(out_seq3_x.size(0), -1)
        out_fc1_x = self.fc_layer1(flattened_x)
        out_fc2_x = self.fc_layer2(out_fc1_x)
        return out_fc2_x


def my_loss_function(prediction, target):
    loss = F.cross_entropy(prediction, target)
    return loss
