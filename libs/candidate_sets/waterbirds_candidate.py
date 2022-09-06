from torch.utils.data import DataLoader
from torchvision import transforms

from libs.utils.wilds_utils import WILDS_utils
from .transformation import Segment_Image
from .CustomCompose import MyCompose
from .wilds_candidate import WILDS_Candidate

import numpy as np

dataset_name = "waterbirds"

class Waterbirds_Candidate_Set(WILDS_Candidate):
    def __init__(self, reshape_size, batch_size):
        super(Waterbirds_Candidate_Set, self).__init__(dataset_name, reshape_size, batch_size)
        self.batch_size = batch_size
        self.reshape_size = (reshape_size, reshape_size)
    
    def get_loader_dict(self):
        return {
            'segment': self.get_train_loader_segment_default(),
            'orig': super().get_train_loader_orig(),
            'segment_2': self.get_train_loader_segment_2(),
        }
    
    def get_train_loader_orig(self):
        trainloader_orig = super().get_train_loader_orig()
        return trainloader_orig
    
    def get_train_loader_segment_default(self):
        _ , trainloader_segment = super().get_train_dataloader(
            self.batch_size, 
            MyCompose(
                [transforms.Resize(self.reshape_size), Segment_Image(), transforms.ToTensor()]
            ),)
        return trainloader_segment
    
    def get_train_loader_segment_2(self):
        _, trainloader_segment_2 = super().get_train_dataloader(
            self.batch_size, 
            MyCompose(
                [transforms.Resize(self.reshape_size), Segment_Image(model='fcn'), transforms.ToTensor()]
            ),)
        return trainloader_segment_2
    
    # def get_test_loader(self):
    #     return super().get_test_loader()

    # def get_val_loader(self):
    #     return super().get_val_loader()
    
    # def get_train_metadata(self):
    #     return super().get_train_metadata()
    
    # def get_test_metadata(self):
    #     return super().get_test_metadata()
    
    # def get_val_metadata(self):
    #     return super().get_val_metadata()