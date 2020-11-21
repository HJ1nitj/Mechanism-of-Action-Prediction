import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

num_input_features = 3+772+100
num_output_features = 206
batch_size = 256
num_epochs = 20

class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # img_size=16x16
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # img_size=8x8
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),# img_size=4x4,
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),# img_size=4x4      
        )
        self.fc_layer = nn.Sequential(
            
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(in_features=512, out_features=num_output_features)  
        
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128*4*4)
        x  = self.fc_layer(x)
        return x
