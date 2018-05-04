"""
ball_model_viz.py

Assuming we have a pretrained model in model.pt, loads in a pretrained model and
performs additional visualizations, creating plots in folder plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using CUDA")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def generate_data(n, theta, v, var_z):
    """
    Generates n noisy data points (x1,x2,...,xn) at t=1.0,1.0+dt,...,1.0+n*dt for a ball
    thrown at an angle theta from the horizontal at an initial velocity v.
    @param theta - angle in radians from the horiontal
    @param v - initial velocity in m/s
    @param var_z - variance of the scalar noise
    """
    dt = 0.001
    g = 9.81
    X = np.zeros((n, 2))
    for i in range(0, n):
        t = i*dt
        x1_t = v*np.cos(theta)*t
        x2_t = v*np.sin(theta)*t - 0.5*g*t**2
        X[i, 0] = x1_t + np.random.normal(0, var_z)
        X[i, 1] = x2_t + np.random.normal(0, var_z)
    return X

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.d = 50
        self.lstm1 = nn.LSTMCell(2, self.d)
        self.lstm2 = nn.LSTMCell(self.d, self.d)
        self.linear = nn.Linear(self.d, 2)

    def forward(self, input, future = 0):
        outputs = []
        # input.size(0) should be the batch size
        h_t = Variable(torch.zeros(input.size(0), self.d).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), self.d).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), self.d).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), self.d).double(), requires_grad=False)

        #print(input.shape)
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            input_t = input_t.squeeze(0) # do not squeeze for single sample inference!
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == "__main__":
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # Load the best saved model.
    with open("model.pt", 'rb') as f:
        seq = torch.load(f)

    # generate a single trajectory and take different k-mers of it.
    seq_length = 2000
    noise_var = 0.0

    trajectories = []
    trajectories.append(generate_data(seq_length, np.pi/3, 15.0, noise_var))

    np.random.seed(0)

    histories = [trajectory[:-1] for trajectory in trajectories]
    next_vals = [trajectory[1:] for trajectory in trajectories]

    X = np.array(np.reshape(histories, (-1, seq_length - 1, 2)))
    Y = np.array(np.reshape(next_vals, (-1, seq_length - 1, 2)))
    print(Y.shape)

    # try to predict the entire trajectory with different amounts of input data.
    criterion = nn.MSELoss()
    target = Variable(torch.from_numpy(Y), requires_grad=False)
    target = target.cuda()
    losses = []
    for input_length in range(50, seq_length, 50):
        input_data = X[:,:input_length,:]
        input = Variable(torch.from_numpy(input_data), requires_grad=False)
        input = input.cuda()
        print(input.size())

        # begin to predict
        future = seq_length - input_length - 1
        pred = seq(input, future = future)
        print('k-mer fraction: {}/{}'.format(input_length, seq_length))
        loss = criterion(pred, target)
        print('total loss:', loss.cpu().data.numpy())
        losses.append(loss)

        y = pred.cpu().data.numpy()[0]
        x = input.cpu().data.numpy()[0]
        # draw the result
        plt.figure(figsize=(30,10))
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.plot(y[:, 0], y[:, 1], 'r', linewidth = 2.0, label = "Future prediction")
        plt.plot(x[:,0], x[:, 1], 'g', linewidth = 2.0, label = "Ground truth trajectory")
        plt.legend()
        plt.savefig('plots/viz_interp{}.png'.format(future))
        plt.close()
    plt.figure()
    plt.xlabel('Given ground-truth sequence length')
    plt.ylabel('Total loss')
    plt.plot(list(range(50, seq_length, 50)), losses)
    plt.savefig('plots/viz_total_loss_interp.png')
    plt.close()
