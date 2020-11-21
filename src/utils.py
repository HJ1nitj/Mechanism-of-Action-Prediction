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

def evaluate(model, validloader, criterion):
    final_loss = 0
    valid_preds = []
    for i, data in enumerate(validloader):
        features, labels = data
        outputs = model(features)
        loss = criterion(outputs.float(), labels.float())
        final_loss+=loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
    
    final_loss /= len(validloader)
    valid_preds = np.concatenate(valid_preds)
    
    return final_loss, valid_preds

def train(model, trainloader, criterion, optimizer):
    running_loss = 0
    for i, data in enumerate(trainloader):
        features, labels = data
        optimizer.zero_grad()
        outputs = model(features)
        #loss
        loss = criterion(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

    training_loss = running_loss/len(trainloader)
    return training_loss, model

def test_prediction(model, testloader):
    preds_df = pd.DataFrame()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            features = data
            outputs = model(features)
            temp_df = pd.DataFrame(F.sigmoid(outputs).numpy())
            preds_df = pd.concat([preds_df, temp_df])
    return preds_df

def rescale_cartersian_cords(arr, img_size):
    arr_x = arr[:, 0].reshape((arr.shape[0], 1))
    arr_y = arr[:, 1].reshape((arr.shape[0], 1))
    new_arr_x = ((arr_x - arr_x.min()) * (1/(arr_x.max() - arr_x.min()) * img_size)).astype('uint8')
    new_arr_y = ((arr_y - arr_y.min()) * (1/(arr_y.max() - arr_y.min()) * img_size)).astype('uint8')
    rescaled_arr = np.concatenate((new_arr_x, new_arr_y), axis=1)
    return rescaled_arr