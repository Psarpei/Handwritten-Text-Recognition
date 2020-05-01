import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

NUM_FEATURES_CNN = 64
LSTM_NUM_DIRS = 2

class CNN_BiLSTM(nn.Module):


    def __init__(self, chars, batch_size, device, max_text_len=1024,
              n_hidden=128, num_layers=3, drop_prob=0.5):
        super(CNN_BiLSTM, self).__init__()
        self.chars = chars
        self.num_chars = len(chars)
        self.batch_size = batch_size
        self.device = device
        # setup CNN
        self.conv1 = nn.Conv2d(1, 2, 5, padding=2)  # input features, output features, kernel size
        self.mp1 = nn.MaxPool2d((2,1), (2,1))  # kernel size, stride --> no overlap!
        self.bn1 = nn.BatchNorm2d(2, affine=False)

        self.conv2 = nn.Conv2d(2, 4, 5, padding=2)  # input features, output features, kernel size
        self.mp2 = nn.MaxPool2d((2,1), (2,1))
        self.bn2 = nn.BatchNorm2d(4, affine=False)

        self.conv3 = nn.Conv2d(4, 8, 3, padding=1)  # input features, output features, kernel size
        self.mp3 = nn.MaxPool2d((2,1), (2,1)) # only max pool vertically from here on
        self.bn3 = nn.BatchNorm2d(8, affine=False)

        self.conv4 = nn.Conv2d(8, 16, 3, padding=1)  # input features, output features, kernel size
        self.mp4 = nn.MaxPool2d((2,1), (2,1))
        self.bn4 = nn.BatchNorm2d(16, affine=False)

        self.conv5 = nn.Conv2d(16, 32 ,3, padding=1)
        self.mp5 = nn.MaxPool2d((2,1), (2,1))
        self.bn5 = nn.BatchNorm2d(32, affine=False)

        self.conv6 = nn.Conv2d(32, NUM_FEATURES_CNN, 3, padding=1)
        self.mp6 = nn.MaxPool2d((2,1), (2,1))
        self.bn6 = nn.BatchNorm2d(64)

        #self.conv7 = nn.Conv2d(64, 128, 3, padding=1)
        #self.mp7 = nn.MaxPool2d((2,1), (2,1))
        #self.bn7 = nn.BatchNorm2d(128)

        # setup BiLSTM
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = NUM_FEATURES_CNN,
                            hidden_size = n_hidden,
                            num_layers = num_layers,
                            bias= True,
                            dropout = drop_prob,
                            bidirectional = True)


        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden*LSTM_NUM_DIRS, self.num_chars+1)

    def init_hidden(self, batch_size):
        weight = next(self.lstm.parameters()).data



        hidden = (Variable(torch.full((self.num_layers*LSTM_NUM_DIRS, batch_size, self.n_hidden), 50.).to(self.device)),
                    Variable(torch.full((self.num_layers*LSTM_NUM_DIRS, batch_size, self.n_hidden), 50.).to(self.device)))
        #hidden = Variable(torch.zeros(self.num_layers*LSTM_NUM_DIRS, batch_size, self.n_hidden).to(self.device))
        return hidden

    def cnn_only(self, x):

        x = self.mp1(F.relu(self.bn1(self.conv1(x))))
        #print(x.shape)
        x = self.mp2(F.relu(self.bn2(self.conv2(x))))
        #print(x.shape)
        x = self.mp3(F.relu(self.bn3(self.conv3(x))))
        #print(x.shape)
        x = self.mp4(F.relu(self.bn4(self.conv4(x))))

        x = self.mp5(F.relu(self.bn5(self.conv5(x))))

        x = self.mp6(F.relu(self.bn6(self.conv6(x))))

        return x
    def forward(self, x):

        # pass the input throug the CNN-part
        #print(x.shape)


        hidden = self.init_hidden(x.size(0))
        x = self.mp1(F.relu(self.bn1(self.conv1(x))))
        #print(x.shape)
        x = self.mp2(F.relu(self.bn2(self.conv2(x))))
        #print(x.shape)
        x = self.mp3(F.relu(self.bn3(self.conv3(x))))
        #print(x.shape)
        x = self.mp4(F.relu(self.bn4(self.conv4(x))))

        x = self.mp5(F.relu(self.bn5(self.conv5(x))))

        x = self.mp6(F.relu(self.bn6(self.conv6(x))))
        #x = self.mp7(F.relu(self.bn7(self.conv7(x))))
        #print(x.shape)

         # current shape of x
        # (batch_size, features, height, time_steps)
        # expects shape (seq_len=time_steps, batch_size, input_size)

        # first: permute the shape to
        # (time_steps, batch_size, features, height)
        # then merge the features and height
        x = x.permute(3, 0, 1, 2)
        x = x.contiguous().view(x.shape[0], x.shape[1], -1)

        #print(x.shape)
        # pass through LSTM with a fresh hidden layer

        #x, hidden = self.rnn(x, hidden)
        x, hidden = self.lstm(x, hidden)

        # dropout_layer
        x = self.dropout(x)

        # pass through fully connected layer to get values for letters

        x = self.fc(x)

        # CTC-loss needs log-probs, therefore log_softmax

        x = F.log_softmax(x, dim=2)

        return x

    def prediction_to_string(self, prediction):
        """Get the string out of the log_probs-output from the network.
        params:
            prediction: log_probs of shape (batch_size=1, seq_len, num_classes)

        out: string with the most probable characters per time_step, with
             repeating characters and blanks removed
        """
        seq = []
        # get the integer-representation for the most likeliest characters
        for i in range(prediction.shape[1]):
            label = np.argmax(prediction[0][i])
            seq.append(label - 1)
        # remove doubles
        #print(seq)
        out = []
        for i in range(len(seq)):
            if len(out) == 0:
                if seq[i] != -1:
                    out.append(seq[i])
            else:
                if seq[i] != -1 and seq[i] != seq[i - 1]: # remove doubles and blanks
                    out.append(seq[i])

        out = "".join(self.chars[i] for i in out)
        return out

    def prediction_to_raw_string(self, prediction):
        """Get the string out of the log_probs-output from the network.
        params:
            prediction: log_probs of shape (batch_size=1, seq_len, num_classes)

        out: string with the most probable characters per time_step, with
             repeating characters and blanks removed
        """
        seq = []
        # get the integer-representation for the most likeliest characters
        for i in range(prediction.shape[1]):
            label = np.argmax(prediction[0][i])
            seq.append(label - 1)
        # remove doubles
        #print(seq)
        out = []
        for i in range(len(seq)):
            if seq[i] == -1:
                out.append("|")
            else:
                out.append(self.chars[seq[i]])

        out = "".join(out)
        return out


        out = "".join( [i for i in out])
        return out


    def decode(self, predictions):
        # prediction is of shape (time_steps, batch_size, output_size=num_chars+1)
        # permute to (batch_size, time_steps, output_size) and cast to numpy
        predictions = predictions.permute(1,0,2).cpu().data.numpy()
        # get the string for each item in the batch
        seq = []
        for i in range(predictions.shape[0]):
            # expects shape (1, time_steps, num_chars)
            pred = predictions[i].reshape(1, predictions.shape[1], predictions.shape[2])
            seq.append(self.prediction_to_string(pred))

        return seq
