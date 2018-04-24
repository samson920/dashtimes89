import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


def generate_data():
    """
    Takes raw genomic data and returns an len(seq)/3 x 1 array of codons, multiple genomes found here:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2974192/
    """
    with open('E_coli_K-12_MG1655_U000962.txt', 'r') as myfile:
        raw_data = myfile.read().replace('\n', '')
    codon_data = []
    for i in range(0,int(np.floor(len(raw_data)-1)/3)):
        codon_data += [raw_data[(3*i):(3*i+3)]]
    print(len(codon_data))
    return codon_data.reshape(len(codon_data),1)

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.d = 2
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
            input_t = input_t.squeeze()
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
    data = generate_data()
    seq_length = len(data)
    noise_var = 0.0

    np.random.seed(0)

    histories = [trajectory[:-1] for trajectory in trajectories]
    next_vals = [trajectory[1:] for trajectory in trajectories]


    X = np.array(np.reshape(histories, (-1, seq_length - 1, 2)))
    num_train = np.floor(seq_length*0.7)
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
    test_input = Variable(torch.from_numpy(testX), requires_grad=False)
    test_target = Variable(torch.from_numpy(testY), requires_grad=False)
    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    #begin to train

    EPOCHS = 40

    for i in range(EPOCHS):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.data.numpy()[0])
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict
        future = 1
        pred = seq(test_input, future = future)
        loss = criterion(pred[:, :-future], test_target)
        print('test loss:', loss.data.numpy()[0])
        y = pred.data.numpy()
        x = test_input.data.numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(yi[:-1, 0], yi[:-1, 1], color, linewidth = 2.0)
            #plt.plot(yi[-1,0], yi[-1:1], color + ':', linewidth = 2.0)
        draw(y[0], 'r:')
        draw(x[0], 'r')
        draw(y[1], 'g:')
        draw(x[1], 'g')
        draw(y[2], 'b:')
        draw(x[2], 'b')
        plt.savefig('predict{}.pdf'.format(i))
        plt.close()
