from libs.domainbed import datasets
from libs.candidate_sets.utils import domainbed_const

import numpy as np

class DomainBed_utils:
    def __init__(self, dataset_name):
        self.dataset = vars(datasets)[dataset_name](domainbed_const.DATA_DIR,
                                               domainbed_const.TEST_ENVS, {'data_augmentation': None})
        self.num_envs = len(self.dataset)
        self.test_envs = domainbed_const.TEST_ENVS
        self.train_envs = [i for i in range(self.num_envs) if i not in self.test_envs]

    def calc_acc_per_group(self, outputs, labels, group_idxs):
        group_outputs = outputs[group_idxs]
        group_labels = labels[group_idxs]
        return np.argwhere(group_outputs == group_labels).shape[0] / len(group_labels)
    
    def get_key_str(self, key):
        key_str = {
            'avg_acc_all': "Avg acc all anvs",
            'avg_acc_test_envs_all': "Avg acc test envs",
            "acc_wg": "Worst group acc"
        }
        if "acc_env_" not in key:
            return key_str[key]
        else:
            return f"acc {key[4:]}"
    
    def get_results_str(self, results):
        return_str = ""
        for key in results:
            return_str += self.get_key_str(key) + ": {:.3f}".format(results[key]) + "\n"
        return return_str
    
    def evaluate_domainbed(self, outputs, labels, metadata):
        '''
        Calculate: 
        1. average accuracy on all envs
        3. average accuracy on all test envs
        4. accuracy on each env
        '''
        results = {}
        outputs = outputs.detach().cpu().numpy().flatten()
        labels = labels.detach().cpu().numpy().flatten()
        metadata = metadata.detach().cpu().numpy().flatten()
        
        if np.unique(metadata).shape[0] > 1:
            avg_acc_all = np.argwhere(outputs == labels).shape[0] / len(labels)
            results['avg_acc_all'] = avg_acc_all
        
        test_env_point_idxs = np.argwhere(np.isin(metadata, self.test_envs)).ravel()

        if len(test_env_point_idxs) > 0:
            avg_acc_test_envs_all = self.calc_acc_per_group(outputs, labels, test_env_point_idxs)
            results['avg_acc_test_envs_all'] = avg_acc_test_envs_all
        
        min_env_acc = float('inf')
        for env_i in range(self.num_envs):
            env_point_idxs = np.argwhere(metadata == env_i)
            if len(env_point_idxs) == 0:
                continue
            acc_test_env = self.calc_acc_per_group(outputs, labels, env_point_idxs)
            results[f'acc_env_{env_i}'] = acc_test_env
            if acc_test_env < min_env_acc:
                min_env_acc = acc_test_env
        
        results['acc_wg'] = min_env_acc
        results['acc_wg'] = np.amin(np.array(list(results.values())))
        
        results_str = self.get_results_str(results)
        return results, results_str
        
        # np.argwhere(labels.flatten() in self.train_envs)
        # print(avg_acc_all)
        # print(train_env_point_idxs)
        # exit()
        
        