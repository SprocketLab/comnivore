from libs.domainbed import datasets
from .CustomCompose import MyCompose
from .utils import domainbed_const
from .transformation import Change_cmap, Convert_to_BW

from .domainbed_candidate import DomainBed_Candidate_Set


class ColoredMNIST_Candidate_Set(DomainBed_Candidate_Set):
    def __init__(self, reshape_size, batch_size):
        self.dataset_orig = vars(datasets)["ColoredMNIST"](domainbed_const.DATA_DIR,
                                               domainbed_const.TEST_ENVS, {'data_augmentation': None})

        super(ColoredMNIST_Candidate_Set, self).__init__(self.dataset_orig)
        self.batch_size = batch_size
        self.reshape_size = (reshape_size, reshape_size)
    
    def get_loader_dict(self):
        return {
            # 'cmap': self.get_train_loader_cmap(),
            'orig': self.get_train_loader_orig(),
            'bw': self.get_train_loader_bw(),
        }
    
    def get_train_dataloader(self, dataset, batch_size):
        train_loader = super().get_mnist_train_loader(dataset, batch_size)
        return train_loader
    
    def get_val_dataloader(self, dataset, batch_size):
        val_loader = super().get_mnist_val_loader(dataset, batch_size)
        return val_loader
    
    def get_test_dataloader(self, dataset, batch_size):
        test_loader = super().get_mnist_test_loader(dataset, batch_size)
        return test_loader
    
    def get_train_loader_orig(self):
        trainloader_orig = self.get_train_dataloader(
            self.dataset_orig,
            self.batch_size
        )
        return trainloader_orig
    
    def get_train_loader_cmap(self):
        extra_transform = MyCompose(
            [Change_cmap()]
        )
        dataset = vars(datasets)["ColoredMNIST"](domainbed_const.DATA_DIR,
                                               domainbed_const.TEST_ENVS, {'data_augmentation': None}, extra_transform)
        # super().set_dataset(dataset)
        trainloader_cmap = self.get_train_dataloader(
            dataset,
            self.batch_size,
        )
        # super().set_dataset(self.dataset)
        return trainloader_cmap
    
    def get_train_loader_bw(self):
        extra_transform = MyCompose(
            [Convert_to_BW()]
        )
        dataset = vars(datasets)["ColoredMNIST"](domainbed_const.DATA_DIR,
                                               domainbed_const.TEST_ENVS, {'data_augmentation': None}, extra_transform)
        # super().set_dataset(dataset)
        trainloader_bw = self.get_train_dataloader(
            dataset,
            self.batch_size,
            
        )
        # super().set_dataset(self.dataset)
        return trainloader_bw
    
    def get_test_loader(self):
        test_loader = super().get_mnist_test_loader(
            self.dataset_orig,
            self.batch_size
        )
        return test_loader

    def get_val_loader(self):
        valloader = self.get_val_dataloader(
            self.dataset_orig,
            self.batch_size
        )
        return valloader
    
    def get_test_metadata(self):
        return super().get_test_metadata()
    
    def get_val_metadata(self):
        return super().get_val_metadata()
    
    def get_train_metadata(self):
        return super().get_train_metadata()
        