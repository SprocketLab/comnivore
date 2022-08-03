from libs.domainbed import datasets
from .transformation import Segment_Image
from .CustomCompose import MyCompose
from .utils import domainbed_const

from .domainbed_candidate import DomainBed_Candidate_Set

dataset = vars(datasets)["OfficeHome"](domainbed_const.DATA_DIR,
                                               domainbed_const.TEST_ENVS, {'data_augmentation': None})


class OfficeHome_Candidate_Set(DomainBed_Candidate_Set):
    def __init__(self, reshape_size, batch_size):
        super(OfficeHome_Candidate_Set, self).__init__(dataset)
        self.batch_size = batch_size
        self.reshape_size = (reshape_size, reshape_size)
    
    def get_loader_dict(self):
        return {
            'segment': self.get_train_loader_segment_default(),
            'orig': self.get_train_loader_orig(),
            'segment_2': self.get_train_loader_segment_2(),
        }
    
    def get_train_dataloader(self, batch_size, transform=None):
        train_loader = super().get_train_loader(batch_size, transform)
        return train_loader
    
    def get_val_dataloader(self, batch_size):
        val_loader = super().get_val_loader(batch_size)
        return val_loader
    
    def get_train_loader_orig(self):
        trainloader_orig = self.get_train_dataloader(self.batch_size)
        return trainloader_orig
    
    def get_train_loader_segment_default(self):
        trainloader_segment = self.get_train_dataloader(
            self.batch_size, 
            MyCompose(
                [Segment_Image()]
            ),)
        return trainloader_segment
    
    def get_train_loader_segment_2(self):
        trainloader_segment_2 = self.get_train_dataloader(
            self.batch_size, 
            MyCompose(
                [Segment_Image(model='fcn')]
            ),)
        return trainloader_segment_2
    
    def get_test_loader(self):
        return None

    def get_val_loader(self):
        valloader = self.get_val_dataloader(self.batch_size)
        return valloader
    
    def get_test_metadata(self):
        return None
    
    def get_val_metadata(self):
        return None
    
    def get_train_metadata(self):
        return None
        