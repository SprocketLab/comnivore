from torch.utils.data import DataLoader
from torchvision import transforms

from libs.utils.wilds_utils import WILDS_utils
from .wilds_candidate import WILDS_Candidate

import numpy as np

dataset_name = "iwildcam"
# dataset = WILDS_utils(dataset_name).dataset

class IWildCam_Candidate_Set(WILDS_Candidate):
    def __init__(self, reshape_size, batch_size):
        super(IWildCam_Candidate_Set, self).__init__(dataset_name, reshape_size, batch_size)
        self.batch_size = batch_size
        self.reshape_size = (reshape_size, reshape_size)
    
    def get_loader_dict(self):
        return {
            'orig': super().get_train_loader_orig(),
            'augment': self.get_train_loader_augment(),
            'position': self.get_train_loader_position(),
        }
    
    def get_train_loader_augment(self):
        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _, trainloader_augment = super().get_train_dataloader(
            self.batch_size, 
            augment_transform,)
        return trainloader_augment
    
    def get_train_loader_position(self):
        position_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            transforms.ToTensor(),
        ])
        _, trainloader_position = super().get_train_dataloader(
            self.batch_size, 
            position_transform,)
        return trainloader_position
    
    def get_test_loader(self):
        return super().get_test_loader()

    def get_val_loader(self):
        return super().get_val_loader()
    
    def get_train_metadata(self):
        return super().get_train_metadata()
    
    def get_test_metadata(self):
        return super().get_test_metadata()
    
    def get_val_metadata(self):
        return super().get_val_metadata()