from libs.utils.wilds_utils import WILDS_utils
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

class WILDS_Candidate:
    def __init__(self, dataset_name, reshape_size, batch_size):
        self.dataset = WILDS_utils(dataset_name).dataset
        self.reshape_size = (reshape_size,reshape_size)
        self.batch_size = batch_size
    
    def get_train_dataloader(self, batch_size, transform):
        dataset_ = self.dataset.get_subset(split="train", transform=transform)
        return dataset_, DataLoader(dataset_, batch_size=batch_size,
                            shuffle=False)
    
    def get_metadata(self, loader):
        metadata_all = []
        for _, (_, _, metadata) in enumerate(loader):
            metadata_all.append(metadata)
        metadata_all = np.vstack(metadata_all)
        return metadata_all
    
    def get_train_loader_orig(self):
        traindata_orig = self.dataset.get_subset(split="train", transform=transforms.Compose(
                [transforms.Resize(self.reshape_size), transforms.ToTensor()]))
        trainloader_orig = DataLoader(traindata_orig, batch_size=self.batch_size,
                                shuffle=False)
        return trainloader_orig
    
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
    
    def get_test_loader(self):
        test_data = self.dataset.get_subset(
            "test",
            transform=transforms.Compose(
                [transforms.Resize(self.reshape_size), transforms.ToTensor()]
            ),
        )
        testloader = DataLoader(test_data, batch_size=self.batch_size,
                                shuffle=False)
        return testloader

    def get_val_loader(self):
        val_data = self.dataset.get_subset(
            "val",
            transform=transforms.Compose(
                [transforms.Resize(self.reshape_size), transforms.ToTensor()]
            ),
        )
        valloader = DataLoader(val_data, batch_size=self.batch_size,
                                shuffle=False)
        return valloader
