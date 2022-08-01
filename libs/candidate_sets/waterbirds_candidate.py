from torch.utils.data import DataLoader
from torchvision import transforms

from libs.utils.wilds_utils import WILDS_utils
from .transformation import Segment_Image
from .CustomCompose import MyCompose

import numpy as np

dataset_name = "waterbirds"
dataset = WILDS_utils(dataset_name).dataset

class Waterbirds_Candidate_Set:
    def __init__(self, reshape_size, batch_size):
        self.batch_size = batch_size
        self.reshape_size = (reshape_size, reshape_size)
    
    def get_loader_dict(self):
        return {
            'segment': self.get_train_loader_segment_default(),
            'orig': self.get_train_loader_orig(),
            'segment_2': self.get_train_loader_segment_2(),
        }

    def get_train_dataloader(self, batch_size, transform):
        dataset_ = dataset.get_subset(split="train", transform=transform)
        return dataset_, DataLoader(dataset_, batch_size=batch_size,
                            shuffle=False)
    
    def get_train_loader_orig(self):
        traindata_orig = dataset.get_subset(split="train", transform=transforms.Compose(
                [transforms.Resize(self.reshape_size), transforms.ToTensor()]))
        trainloader_orig = DataLoader(traindata_orig, batch_size=self.batch_size,
                                shuffle=False)
        return trainloader_orig
    
    def get_train_loader_segment_default(self):
        _ , trainloader_segment = self.get_train_dataloader(
            self.batch_size, 
            MyCompose(
                [transforms.Resize(self.reshape_size), Segment_Image(), transforms.ToTensor()]
            ),)
        return trainloader_segment
    
    def get_train_loader_segment_2(self):
        _, trainloader_segment_2 = self.get_train_dataloader(
            self.batch_size, 
            MyCompose(
                [transforms.Resize(self.reshape_size), Segment_Image(model='fcn'), transforms.ToTensor()]
            ),)
        return trainloader_segment_2
    
    def get_test_loader(self):
        test_data = dataset.get_subset(
            "test",
            transform=transforms.Compose(
                [transforms.Resize(self.reshape_size), transforms.ToTensor()]
            ),
        )
        testloader = DataLoader(test_data, batch_size=self.batch_size,
                                shuffle=False)
        return testloader

    def get_val_loader(self):
        val_data = dataset.get_subset(
            "val",
            transform=transforms.Compose(
                [transforms.Resize(self.reshape_size), transforms.ToTensor()]
            ),
        )
        valloader = DataLoader(val_data, batch_size=self.batch_size,
                                shuffle=False)
        return valloader

    def get_metadata(self, loader):
        metadata_all = []
        for _, (_, _, metadata) in enumerate(loader):
            metadata_all.append(metadata)
        metadata_all = np.vstack(metadata_all)
        return metadata_all
    
    def get_train_metadata(self):
        trainloader_orig = self.get_train_loader_orig()
        train_metadata = self.get_metadata(trainloader_orig)
        return train_metadata
    
    def get_test_metadata(self):
        testloader = self.get_test_loader()
        test_metadata = self.get_metadata(testloader)
        return test_metadata
    
    def get_val_metadata(self):
        valloader = self.get_val_loader()
        val_metadata = self.get_metadata(valloader)
        return val_metadata