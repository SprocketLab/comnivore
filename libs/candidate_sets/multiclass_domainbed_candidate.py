from libs.domainbed import datasets
from .transformation import Segment_Image
from .CustomCompose import MyCompose
from .utils import domainbed_const

from .domainbed_candidate import DomainBed_Candidate_Set



class MultiClassDomainBed(DomainBed_Candidate_Set):
    def __init__(self, reshape_size, batch_size, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = vars(datasets)[dataset_name](domainbed_const.DATA_DIR,
                                               domainbed_const.TEST_ENVS, {'data_augmentation': False})
        super(MultiClassDomainBed, self).__init__(self.dataset, dataset_name)
        self.batch_size = batch_size
        self.reshape_size = (reshape_size, reshape_size)
    
    def get_loader_dict(self):
        return {
            'segment': self.get_train_loader_segment_default(),
            'orig': self.get_train_loader_orig(),
            'augment': self.get_train_loader_default_augment(),
            'segment_2': self.get_train_loader_segment_2(),
        }
    
    def get_train_dataloader(self, dataset, batch_size):
        train_loader = super().get_train_loader(dataset, batch_size)
        return train_loader
    
    def get_val_dataloader(self, dataset, batch_size):
        val_loader = super().get_val_loader(dataset, batch_size)
        return val_loader
    
    def get_test_dataloader(self, dataset, batch_size):
        test_loader = super().get_test_loader(dataset, batch_size)
        return test_loader
    
    def get_train_loader_orig(self):
        trainloader_orig = self.get_train_dataloader(
            self.dataset,
            self.batch_size
        )
        return trainloader_orig
    
    def get_train_loader_segment_default(self):
        extra_transforms = MyCompose(
            [Segment_Image()]
        )
        dataset = vars(datasets)[self.dataset_name](domainbed_const.DATA_DIR,
                                               domainbed_const.TEST_ENVS, {'data_augmentation': False}, extra_transforms)
        # super().set_dataset(dataset)
        trainloader_segment = self.get_train_dataloader(
            dataset,
            self.batch_size, )
        # super().set_dataset(self.dataset)
        return trainloader_segment
    
    def get_train_loader_default_augment(self):
        dataset = vars(datasets)[self.dataset_name](domainbed_const.DATA_DIR,
                                               domainbed_const.TEST_ENVS, {'data_augmentation': True})
        # super().set_dataset(dataset)
        trainloader_segment_2 = self.get_train_dataloader(
            dataset,
            self.batch_size,)
        # super().set_dataset(self.dataset)
        return trainloader_segment_2
    
    def get_train_loader_segment_2(self):
        extra_transforms = MyCompose(
            [Segment_Image(model='fcn')]
        )
        dataset = vars(datasets)[self.dataset_name](domainbed_const.DATA_DIR,
                                               domainbed_const.TEST_ENVS, {'data_augmentation': False}, extra_transforms)
        # super().set_dataset(dataset)
        trainloader_segment_2 = self.get_train_dataloader(
            dataset,
            self.batch_size,)
        # super().set_dataset(self.dataset)
        return trainloader_segment_2
    
    def get_test_loader(self):
        test_loader = super().get_test_loader(self.dataset, self.batch_size)
        return test_loader

    def get_val_loader(self):
        valloader = self.get_val_dataloader(self.dataset, self.batch_size)
        return valloader
    
    def get_test_metadata(self):
        return super().get_test_metadata()
    
    def get_val_metadata(self):
        return super().get_val_metadata()
    
    def get_train_metadata(self):
        return super().get_train_metadata()
        