from .waterbirds_candidate import Waterbirds_Candidate_Set
from .officehome_candidate import OfficeHome_Candidate_Set
from .coloredmnist_candidate import ColoredMNIST_Candidate_Set
from libs.model import *
from tqdm import tqdm


class Candidate_Set:
    def __init__(self, dataset_name, reshape_size, batch_size):
        self.dataset_name = dataset_name
        self.reshape_size = reshape_size
        self.batch_size = batch_size
        self.candidate_set = self.get_candidate_set(dataset_name)
        
    def get_candidate_set(self, dataset_name):
        if dataset_name == 'waterbirds':
            return Waterbirds_Candidate_Set(self.reshape_size, self.batch_size)
        if dataset_name == 'OfficeHome':
            return OfficeHome_Candidate_Set(self.reshape_size, self.batch_size)
        if dataset_name == 'ColoredMNIST':
            return ColoredMNIST_Candidate_Set(self.reshape_size, self.batch_size)
    
    def get_all_train_loader_by_tasks(self, tasks):
        loaders = []
        log("Getting tain loaders for all environments....")
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