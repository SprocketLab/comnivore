from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os
from PIL import Image
from torchvision import transforms
from .transformation import Convert_to_BW
from .CustomCompose import MyCompose
import numpy as np

class Synthetic_CMNIST_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.file_list = os.listdir(self.root_dir)
        self.transform = transform
        self.df = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.file_list[idx]
        img_path = os.path.join(self.root_dir, img_path)
        
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if type(img) is np.ndarray:
            img = torch.Tensor(img)
        elif not torch.is_tensor(img):
            totensor = transforms.Compose([
                    transforms.ToTensor()])
            img = totensor(img)
        file_in_df = self.df.loc[self.df["image_path"] == img_path]
        label = file_in_df.values[0][2]
        return img, label


class Synthetic_ColoredMNIST_Candidate_Set:
    def __init__(self, batch_size, root_cmnist_path="/hdd2/dyah/coloredmnist_synthetic_spurious"):
        self.batch_size = batch_size
        self.root_cmnist_path = root_cmnist_path
        self.csv_path = os.path.join(self.root_cmnist_path, "metadata.csv")
        self.df = pd.read_csv(self.csv_path)
        
    def get_loader_dict(self):
        return {
            'orig': self.get_train_loader_orig(),
            'bw': self.get_train_loader_bw(),
        }
        
    def get_train_loader_orig(self):
        traindata = Synthetic_CMNIST_Dataset(self.csv_path, os.path.join(self.root_cmnist_path, "train"))
        trainloader_orig = DataLoader(traindata, batch_size=self.batch_size)
        return trainloader_orig
    
    def get_train_loader_bw(self):
        extra_transform = MyCompose(
            [Convert_to_BW()]
        )
        traindata_bw = Synthetic_CMNIST_Dataset(self.csv_path, os.path.join(self.root_cmnist_path, "train"), extra_transform)
        trainloader_bw = DataLoader(traindata_bw, batch_size = self.batch_size, shuffle=False)
        return trainloader_bw
    
    def get_test_loader(self):
        testdata = Synthetic_CMNIST_Dataset(self.csv_path, os.path.join(self.root_cmnist_path, "test"))
        testloader = DataLoader(testdata, batch_size = self.batch_size, shuffle=False)
        return testloader
    
    def get_val_loader(self):
        valdata = Synthetic_CMNIST_Dataset(self.csv_path, os.path.join(self.root_cmnist_path, "val"))
        valloader = DataLoader(valdata, batch_size = self.batch_size, shuffle=False)
        return valloader
    
    def get_test_metadata(self):
        test_rows = self.df.loc[self.df["split"] == 1]
        return test_rows['random'].tolist()
    
    def get_val_metadata(self):
        val_rows = self.df.loc[self.df["split"] == 2]
        return val_rows['random'].tolist()
    
    def get_train_metadata(self):
        train_rows = self.df.loc[self.df["split"] == 0]
        return train_rows['random'].tolist()
    
        
        