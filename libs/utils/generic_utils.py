from libs.domainbed import datasets
from libs.candidate_sets.utils import domainbed_const

import numpy as np
import torch

class Generic_utils:
    def __init__(self):
        pass
    
    def get_results_str(self, results):
        return_str = ""
        for key in results:
            return_str += key + ": {:.3f}".format(results[key]) + "\n"
        return return_str
    
    def calc_acc(self, p, y):
        p = torch.Tensor(p)
        y = torch.Tensor(y)
        correct = (p.eq(y).float()).sum().item()
        total = len(y)
        return correct / total
    
    def evaluate(self, outputs, labels, metadata):
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
        