import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from tcn import TemporalConvNet as TCN
#used Bai et al.'s TCN.py as a starting point
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

if __name__ == "__main__":
    seq_length = 2000
    noise_var = 0.0
    #Generate data
    trajectories = []
    for theta in np.linspace(2*np.pi/10, 4*np.pi/10, 20):
        for v in np.linspace(12.0, 15.0, 20):
            trajectories.append(generate_data(seq_length, theta, v, noise_var))
    for trajectory in trajectories:
        plt.plot(trajectory[:,0], trajectory[:,1])

    np.random.seed(0)
    np.random.shuffle(trajectories)

    histories = [trajectory[:-1] for trajectory in trajectories]
    next_vals = [trajectory[1:] for trajectory in trajectories]

    #Split data into training and test set 
    X = np.array(np.reshape(histories, (-1, seq_length - 1, 2)))
    num_train = 300
    trainX = X[:num_train]
    testX = X[num_train:]
    Y = np.array(np.reshape(next_vals, (-1, seq_length - 1, 2)))
    trainY = Y[:num_train]
    testY = Y[num_train:]
    print(trainY.shape)

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    input = Variable(torch.from_numpy(trainX), requires_grad=False)
    target = Variable(torch.from_numpy(trainY), requires_grad=False)
    test_input = Variable(torch.from_numpy(testX), requires_grad=False, volatile=True)
    test_target = Variable(torch.from_numpy(testY), requires_grad=False, volatile=True)
    if use_cuda:
        input = input.cuda()
        target = target.cuda()
        test_input = test_input.cuda()
        test_target = test_target.cuda()
    # Conv1d input format is (batch_size, num_channels, seq_length)
    # so we have to transpose the second and third dimensions of the vars
    input = torch.transpose(input, 1, 2)
    target = torch.transpose(target, 1, 2)
    test_input = torch.transpose(test_input, 1, 2)
    test_target = torch.transpose(test_target, 1, 2)
    print(input.size(), target.size())
    # build the model
    input_size = 2 # dimension of each sequence element
    num_hidden = 8 # num hidden units per layer
    levels = 10 # num layers
    channel_sizes = [num_hidden]*levels
    kernel_size = 8
    #Use the TCN specified in tcn.py
    seq = TCN(input_size, input_size, channel_sizes, kernel_size, dropout=0.0)
    if use_cuda:
        seq.cuda()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.08)
    #begin to train
    best_loss = 1e8

    EPOCHS = 100

    for i in range(EPOCHS):
        print('EPOCH: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            #print(out[0])
            loss = criterion(out, target)
            print('loss:', loss.cpu().data.numpy())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict
        future = 1
        pred = seq(test_input, future = future)
        loss = criterion(pred[:, :, :-future], test_target)
        print('test loss:', loss.cpu().data.numpy())
        # Save the model if the test loss is the best we've seen so far.
        if loss < best_loss:
            with open("ball_tcn.pt", 'wb') as f:
                print('Save model!\n')
                torch.save(seq, f)
            best_loss = loss

        y = pred.cpu().data.numpy()
        x = test_input.cpu().data.numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(yi[0, :], yi[1, :], color, linewidth = 2.0) # notice we transposed the channels and length
        draw(y[0], 'r:')
        draw(x[0], 'r')
        draw(y[1], 'g:')
        draw(x[1], 'g')
        draw(y[2], 'b:')
        draw(x[2], 'b')
        plt.savefig('plots/tcn_predict{}.png'.format(i))
        plt.close()
