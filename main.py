import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm
sys.path.append('./src')
from utils import *
from dataloader import *
from model import *
from convex_hull import *

data_path = "/kaggle/input/lish-moa/"
num_input_features = 3+772+100
num_output_features = 206
batch_size = 256
num_epochs = 20

#########################
# ---- Load Data ------
########################
train_features_data = pd.read_csv(data_path+'train_features.csv')
test_features_data = pd.read_csv(data_path+'test_features.csv')
train_target_scored = pd.read_csv(data_path+'train_targets_scored.csv')
train_feat_targets_data = pd.merge(train_features_data, train_target_scored, on='sig_id', how='left')


##########################
# ---- Pre-Processing-------
##########################
cp_type_mapping = {'trt_cp':0, 'ctl_vehicle':1}
cp_dose_mapping = {'D1':0, 'D2':1}

train_feat_targets_data['cp_type'] = train_feat_targets_data['cp_type'].map(cp_type_mapping)
train_feat_targets_data['cp_dose'] = train_feat_targets_data['cp_dose'].map(cp_dose_mapping)

test_features_data['cp_type'] = test_features_data['cp_type'].map(cp_type_mapping)
test_features_data['cp_dose'] = test_features_data['cp_dose'].map(cp_dose_mapping)


training_features = [col for col in train_features_data.columns if col!='sig_id']
target_labels = [col for col in train_target_scored.columns if col!='sig_id']

gene_expression_ls = [col for col in train_features_data.columns if 'g-' in col]
cell_viability_ls = [col for col in train_features_data.columns if 'c-' in col]

############################################
# ------- Dimensionality Reduciton ---------
############################################
tsne = TSNE(n_components=2)
train_features_data_2d = tsne.fit_transform(train_features_data[gene_expression_ls].T.values)


###########################
# --- Convex Hull --------
###########################
angle, min_bbox1, min_bbox2, min_bbox3, center_point, corner_points = minBoundingRect(train_features_data_2d)
rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                          [math.sin(angle), math.cos(angle)]])
train_features_data_2d_rotated = np.matmul(train_features_data_2d, rotation_matrix)
corner_points_rotated = np.matmul(corner_points, rotation_matrix)

#############################
# ------ Rescaling ---------
#############################
img_size=32
train_features_data_2d_rescaled = rescale_cartersian_cords(train_features_data_2d_rotated, img_size=img_size)

#########################################
# ------ Storing train images --------
#########################################
geneseq_image_data = []
for sig_id, feature_value in tqdm(zip(train_features_data['sig_id'].values, 
                                      train_features_data[gene_expression_ls].values),
                                      total=len(train_features_data)):
    
    pixel_value_arr = map_feature_values(train_features_data_2d_rescaled, feature_value)
    
    geneseq_image_data.append(pixel_value_arr.reshape((1, img_size, img_size)))

########################################
# --------- Stroing test images ---------
########################################
test_geneseq_image_data = []
for sig_id, feature_value in tqdm(zip(test_features_data['sig_id'].values, 
                                      test_features_data[gene_expression_ls].values),
                                      total=len(test_features_data)):
    
    pixel_value_arr_test = map_feature_values(train_features_data_2d_rescaled, feature_value)
    
    test_geneseq_image_data.append(pixel_value_arr_test.reshape((1, img_size, img_size)))

train_feat_targets_data['image_data'] = geneseq_image_data
test_features_data['image_data'] = test_geneseq_image_data


################################################
# ---- Training in Stratified K-Fold Setting
################################################
final_preds_arr = np.zeros((len(test_features_data), len(target_labels)))
kfold = MultilabelStratifiedKFold(n_splits=5, random_state=1008, shuffle=True)

for n, (train_idx, valid_idx) in enumerate(kfold.split(train_feat_targets_data, train_feat_targets_data[target_labels])):
    print ('-'*5)
    print ('Fold {}'.format(n+1))
    print ('-'*5)
    
    # network
    nn_model = cnn_model().float()

    ## loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=nn_model.parameters(), lr=0.001)
    
    # splitting into train and validation
    train_data_fold, valid_data_fold = train_feat_targets_data.loc[train_idx], train_feat_targets_data.loc[valid_idx]
    
    # trainloader
    train_data =  TrainData(train_data_fold, 'image_data', target_labels)
    trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # validloader
    valid_data = TrainData(valid_data_fold, 'image_data', target_labels)
    validloader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
    
    # testloader
    test_data = TestData(test_features_data, 'image_data')
    testloader = DataLoader(dataset=test_data, batch_size=batch_size)
        
    for epoch in range(num_epochs):
        training_loss, nn_model = train(nn_model, trainloader, criterion, optimizer)
        valid_loss, valid_preds = evaluate(nn_model, validloader, criterion)
        print ('\t', 'Epoch {}, Training Loss = {}, Valid Loss = {}'.format(epoch, training_loss, valid_loss))
        
    preds_df = test_prediction(nn_model, testloader).values
    final_preds_arr+=preds_df
final_preds_arr = final_preds_arr/5

final_preds_arr.to_csv('prediction.csv', index=False)