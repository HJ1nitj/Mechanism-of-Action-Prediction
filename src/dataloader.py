import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class TrainData(Dataset):
    def __init__(self, train_features_data, image_data_colname, target_labels):
        self.feat = train_features_data[image_data_colname].values
        self.labels = train_features_data[target_labels].values
        
    def __getitem__(self, idx):
        img = torch.FloatTensor(self.feat[idx])
        target_labels = torch.FloatTensor(self.labels[idx])
        return img, target_labels

    def __len__(self):
        return len(self.feat)
    

class TestData(Dataset):
    def __init__(self, data, image_data_colname):
        self.feat = data[image_data_colname].values
        
    def __getitem__(self, idx):
        img = torch.FloatTensor(self.feat[idx])
        return img

    def __len__(self):
        return len(self.feat)