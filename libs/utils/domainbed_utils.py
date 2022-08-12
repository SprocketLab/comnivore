from libs.domainbed import datasets
from libs.candidate_sets.utils import domainbed_const

import numpy as np
import torch

class DomainBed_utils:
    def __init__(self, dataset_name):
        self.dataset = vars(datasets)[dataset_name](domainbed_const.DATA_DIR,
                                               domainbed_const.TEST_ENVS, {'data_augmentation': None})
        self.num_envs = len(self.dataset)
        self.test_envs = domainbed_const.TEST_ENVS
        self.train_envs = [i for i in range(self.num_envs) if i not in self.test_envs]
    
    def get_results_str(self, results):
        return_str = ""
        for key in results:
            return_str += key + ": {:.3f}".format(results[key]) + "\n"
        return return_str
    
    def calc_acc(self, p, y):
        p = torch.Tensor(p)
        y = torch.Tensor(y)
        # print(p.gt(0))
        correct = (p.eq(y).float()).sum().item()
        total = len(y)
        return correct / total
    
    def evaluate_domainbed(self, outputs, labels, metadata):
        '''
        Calculate accuracy on each env
        '''
        results = {}
        groups = np.unique(metadata)
        min_env_acc = float('inf')
        for group in groups:
            group_idxs = np.argwhere(metadata == group).flatten()
            group_o = outputs[group_idxs]
            group_y = labels[group_idxs]
            group_acc = self.calc_acc(group_o, group_y)
            results[f"{group}_acc"] = group_acc
            if group_acc < min_env_acc:
                min_env_acc = group_acc
        results['acc_wg'] = min_env_acc
        results_str = self.get_results_str(results)
        return results, results_str
        