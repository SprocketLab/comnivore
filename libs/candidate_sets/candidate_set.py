from .waterbirds_candidate import Waterbirds_Candidate_Set
from .multiclass_domainbed_candidate import MultiClassDomainBed
from .coloredmnist_candidate import ColoredMNIST_Candidate_Set
from .iwildcam_candidate import IWildCam_Candidate_Set
from .synthetic_coloredmnist_candidate import Synthetic_ColoredMNIST_Candidate_Set
from libs.model import *
from tqdm import tqdm


class Candidate_Set:
    def __init__(self, dataset_name, reshape_size=(224,224), batch_size=128):
        self.dataset_name = dataset_name
        self.reshape_size = reshape_size
        self.batch_size = batch_size
        self.candidate_set = self.get_candidate_set(dataset_name)
        
    def get_candidate_set(self, dataset_name):
        if dataset_name == 'waterbirds':
            return Waterbirds_Candidate_Set(self.reshape_size, self.batch_size)
        if dataset_name == 'iwildcam':
            return IWildCam_Candidate_Set(self.reshape_size, self.batch_size)
        elif dataset_name == 'OfficeHome':
            return MultiClassDomainBed(self.reshape_size, self.batch_size, 'OfficeHome')
        elif dataset_name == 'VLCS':
            return MultiClassDomainBed(self.reshape_size, self.batch_size, 'VLCS')
        elif dataset_name == 'PACS':
            return MultiClassDomainBed(self.reshape_size, self.batch_size, 'PACS')
        elif dataset_name == 'ColoredMNIST':
            return ColoredMNIST_Candidate_Set(self.batch_size)
        elif dataset_name == "Synthetic_ColoredMNIST":
            return Synthetic_ColoredMNIST_Candidate_Set(self.batch_size)
            
    
    def get_all_train_loader_by_tasks(self, tasks):
        loaders = []
        print("Getting train loaders for all environments....")
        for task in tqdm(tasks):
            task_loader =  self.candidate_set.get_loader_dict()[task]
            loaders.append(task_loader)
        return loaders
    
    def get_test_loader(self):
        return self.candidate_set.get_test_loader()

    def get_val_loader(self):
        return self.candidate_set.get_val_loader()
    
    def get_test_metadata(self):
        return self.candidate_set.get_test_metadata()
    
    def get_val_metadata(self):
        return self.candidate_set.get_val_metadata()
    
    def get_train_metadata(self):
        return self.candidate_set.get_train_metadata()