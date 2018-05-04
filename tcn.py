import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

#TCN starter code taken from Bai et al. (https://arxiv.org/abs/1803.01271), 
#See original code from authors: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.unit_input_length = kernel_size*dilation_size # save the min length required for 1 conv1d output
        self.linear = nn.Linear(num_channels[-1], num_outputs)
        self.linear.weight.data.normal_(0, 0.01)
        print(self.unit_input_length)

    def forward(self, x, future = 0):
        gt_outputs = self.linear(self.network(x).data.transpose(1, 2)) # outputs when fed in ground truth sequence
        gt_outputs = gt_outputs.transpose(1, 2)
        outputs = [gt_outputs]
        last_pred = gt_outputs[:,:,-1].data.unsqueeze(-1)
        # now do future prediction
        # feed in the last output as the last input and perform a single convolution
        # at the end of each layer's sequence to get an additional prediction
        for i in range(future):
            # output is in (batch_size, channels, seq_len) format
            print(x[:,:,-self.unit_input_length:].size(), last_pred.size())
            x = torch.cat((x[:,:,-self.unit_input_length:].data, last_pred.data), 2) # concatenate along seq_len
            last_pred = self.linear(self.network(x).transpose(1, 2))
            last_pred = last_pred.transpose(1, 2)[:,:,-1].unsqueeze(-1)
            outputs.append(last_pred)

        return torch.cat(outputs, 2)
