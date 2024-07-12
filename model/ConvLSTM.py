import torch
import torch as T
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class MODEL(nn.Module):
    def __init__(self,n_feature, n_past, n_future,device):
        super().__init__()
        self.input_size = n_feature
        self.output_size = 1
        self.n_future = n_future
        self.n_past = n_past
        self.hidden_size = 100
        self.num_layers = 1
        self.device = device

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=64, kernel_size=2),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
            
        self.lstm = nn.LSTM(input_size= 128, hidden_size= self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=0.1)         ### input_size: number of expected features
        self.fc = nn.Linear(in_features = (self.n_past-2)//2 * self.hidden_size,out_features = self.n_future) 
        self.flatten = nn.Flatten()



    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        
        x, _ = self.lstm(x, (h0, c0))
        x = self.flatten(x)
        x = self.fc(x)
        x=x.view(x.size(0),-1,self.output_size)
       
        return x



