from libs.candidate_sets.utils.domainbed_const import HOLDOUT_FRACTION
from .utils import domainbed_const
from libs.domainbed.lib import misc
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F


split_seed = domainbed_const.SPLIT_SEED
holdout_fraction = domainbed_const.HOLDOUT_FRACTION
test_envs = domainbed_const.TEST_ENVS

class DomainBed_Candidate_Set:
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_envs = len(dataset)
        self.keys = list(range(len(dataset)))
        np.random.RandomState(split_seed).shuffle(self.keys)
        self.in_splits, self.out_splits = self.get_splits(self.keys)
    
    def set_dataset(self, dataset):
        self.dataset = dataset
        self.in_splits, self.out_splits = self.get_splits(self.keys)
    
    def get_splits(self, keys=None):
        in_splits = []
        out_splits = []
        for env_i, env in enumerate(self.dataset):
            out, in_ = misc.split_dataset(env,
                                            int(len(env) * holdout_fraction),
                                            misc.seed_hash(split_seed, env_i),
                                            keys=keys)
            in_splits.append((in_, env_i))
            out_splits.append((out, env_i))
        return in_splits, out_splits
    
    def add_dataset_dimension(self, dataset):
        pad = torch.zeros((dataset.shape[0], 1, dataset.shape[2], dataset.shape[3]))
        dataset = torch.cat((dataset, pad),1)
        return dataset
    
    def get_mnist_train_loader(self, batch_size):
        _train_envs = [i for i in range(self.num_envs) if i not in test_envs]
        clf_train_features, clf_train_labels = zip(*[self.in_splits[i][0].underlying_dataset.tensors for i in _train_envs])
        clf_train_features, clf_train_labels = torch.cat(clf_train_features), torch.cat(clf_train_labels)
        if clf_train_features.shape[1] < 3:
            clf_train_features = self.add_dataset_dimension(clf_train_features)
        clf_train_dataloader = DataLoader(
            dataset=TensorDataset(clf_train_features, clf_train_labels),
            batch_size=batch_size,)
        return clf_train_dataloader
    
    def get_train_metadata(self):
        _train_envs = [i for i in range(self.num_envs) if i not in test_envs]
        train_metadata = []
        for split in self.in_splits:
            env_i = split[1]
            if isinstance(split[0].underlying_dataset, type(TensorDataset)):
                train_metadata.extend([env_i for i in range(split[0].underlying_dataset.tensors[1].shape[0]) if env_i in _train_envs])
            else:
                train_metadata.extend([env_i for i in range(len(split[0].underlying_dataset.samples)) if env_i in _train_envs])
        return np.vstack(train_metadata)

    def get_val_metadata(self):
        _train_envs = [i for i in range(self.num_envs) if i not in test_envs]
        val_metadata = []
        for split in self.out_splits:
            env_i = split[1]
            if isinstance(split[0].underlying_dataset, type(TensorDataset)):
                val_metadata.extend([env_i for i in range(split[0].underlying_dataset.tensors[1].shape[0]) if env_i in _train_envs])
            else:
                val_metadata.extend([env_i for i in range(len(split[0].underlying_dataset.samples)) if env_i in _train_envs])
        return np.vstack(val_metadata)
    
    def get_test_metadata(self):
        test_metadata = []
        for split in self.in_splits:
            env_i = split[1]
            if isinstance(split[0].underlying_dataset, type(TensorDataset)):
                test_metadata.extend([env_i for i in range(split[0].underlying_dataset.tensors[1].shape[0]) if env_i in test_envs])
            else:
                test_metadata.extend([env_i for i in range(len(split[0].underlying_dataset.samples)) if env_i in test_envs])
        for split in self.out_splits:
            env_i = split[1]
            if isinstance(split[0].underlying_dataset, type(TensorDataset)):
                test_metadata.extend([env_i for i in range(split[0].underlying_dataset.tensors[1].shape[0]) if env_i in test_envs])
            else:
                test_metadata.extend([env_i for i in range(len(split[0].underlying_dataset.samples)) if env_i in test_envs])
        return np.vstack(test_metadata)
    
    def get_train_loader(self, batch_size):
        _train_envs = [i for i in range(self.num_envs) if i not in test_envs]
        train_datasets = [self.in_splits[i][0].underlying_dataset for i in _train_envs]
        train_datasets = torch.utils.data.ConcatDataset(train_datasets)
        train_dataloader = DataLoader(
            dataset=train_datasets,
            batch_size=batch_size,)
        return train_dataloader

    def get_val_loader(self, batch_size):
        _train_envs = [i for i in range(self.num_envs) if i not in test_envs]
        val_datasets = [self.out_splits[i][0].underlying_dataset for i in _train_envs]
        val_datasets = torch.utils.data.ConcatDataset(val_datasets)
        valid_dataloader = DataLoader(
            dataset=val_datasets,
            batch_size=batch_size,)
        return valid_dataloader
    
    def get_test_loader(self, batch_size):
        test_datasets = [self.in_splits[i][0].underlying_dataset for i in test_envs] + \
            [self.out_splits[i][0].underlying_dataset for i in test_envs]
        test_datasets = torch.utils.data.ConcatDataset(test_datasets)
        test_dataloader = DataLoader(
            dataset = test_datasets,
            batch_size=batch_size,
        )
        return test_dataloader

    
    def get_mnist_val_loader(self, batch_size):
        _train_envs = [i for i in range(self.num_envs) if i not in test_envs]
        clf_valid_features, clf_valid_labels = zip(*[self.out_splits[i][0].underlying_dataset.tensors for i in _train_envs])
        clf_valid_features, clf_valid_labels = torch.cat(clf_valid_features), torch.cat(clf_valid_labels)
        if clf_valid_features.shape[1] < 3:
            clf_valid_features = self.add_dataset_dimension(clf_valid_features)
        clf_valid_dataloader = DataLoader(
            dataset=TensorDataset(clf_valid_features, clf_valid_labels),
            batch_size=batch_size,)
        return clf_valid_dataloader
    
    def get_mnist_test_loader(self, batch_size):
        test_splits = [self.in_splits[i][0].underlying_dataset.tensors for i in test_envs] + \
            [self.out_splits[i][0].underlying_dataset.tensors for i in test_envs]
        clf_test_features, clf_test_labels = zip(*test_splits)
        clf_test_features, clf_test_labels = torch.cat(clf_test_features), torch.cat(clf_test_labels)
        if clf_test_features.shape[1] < 3:
            clf_test_features = self.add_dataset_dimension(clf_test_features)
        clf_test_dataloader = DataLoader(
            dataset=TensorDataset(clf_test_features, clf_test_labels),
            batch_size=batch_size,)
        return clf_test_dataloader