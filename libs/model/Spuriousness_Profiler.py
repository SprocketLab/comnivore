import numpy as np
from libs.utils.logger import log

class Spuriousness_Profiler:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    
    def calc_synthethic_cmnist_spuriousness(self, metadata, mode):
        if mode == 'low':
            true_rate = metadata[metadata == 0].shape[0] / metadata.shape[0]
            false_rate = 1. - true_rate
        elif mode == 'high':
            true_rate = metadata[metadata == 1].shape[0] / metadata.shape[0]
            false_rate = 1. - true_rate
        return true_rate, false_rate
    
        # p_spur = metadata[metadata == 0].shape[0] / metadata.shape[0]
        # return p_spur
    
    def calc_waterbirds_spuriousness(self, metadata, mode):
        if mode == 'low':
            true_rate = np.argwhere(metadata[:, 0] == metadata[:, 1]).shape[0] / len(metadata)
            false_rate = 1. - true_rate
        elif mode == 'high':
            true_rate = np.argwhere(metadata[:, 0] != metadata[:, 1]).shape[0] / len(metadata)
            false_rate = 1. - true_rate
        return true_rate, false_rate
    
    def calc_celebA_spuriousness(self, metadata, mode):
        # y (metadata index 1) = 1  : Blonde | cofounder (metadata index 0) = 1 : Male
        # spur_idxs = np.argwhere((metadata[:, 1] ==  0) & (metadata[:, 0] == 1))
        if mode == 'low':
            true_rate = np.argwhere(((metadata[:, 1] ==  1) & (metadata[:, 0] == 0)) | ((metadata[:, 1] ==  0) & (metadata[:, 0] == 1)))
            true_rate = true_rate.shape[0] / len(metadata)
            false_rate = 1.0 - true_rate
        else:
            true_rate = np.argwhere(((metadata[:, 1] ==  1) & (metadata[:, 0] == 1)) | ((metadata[:, 1] ==  0) & (metadata[:, 0] == 0)))
            true_rate = true_rate.shape[0] / len(metadata)
            false_rate = 1.0 - true_rate
        return true_rate, false_rate

    def calc_cmnist_multi_spurious_group(self, metadata, mode):
        log(f"spurious groups % in {mode} weighted samples")
        log(f"% of spurious group 1: {metadata[metadata == 1].shape[0] / metadata.shape[0]}")
        log(f"% of spurious group 2: {metadata[metadata == 2].shape[0] / metadata.shape[0]}")

    def calc_dataset_spuriousness(self, metadata, mode):
        if self.dataset_name == 'Synthetic_ColoredMNIST':
            return self.calc_synthethic_cmnist_spuriousness(metadata, mode)
        elif self.dataset_name == 'Synthetic_ColoredMNIST_Multi':
            self.calc_cmnist_multi_spurious_group(metadata[:, 1], mode)
            return self.calc_synthethic_cmnist_spuriousness(metadata[:, 0], mode)
        elif self.dataset_name == 'waterbirds':
            return self.calc_waterbirds_spuriousness(metadata, mode)
        elif self.dataset_name == 'celebA':
            return self.calc_celebA_spuriousness(metadata, mode)

    def calculate_spuriousness_fix_n(self, metadata, points_weights, n_calc, log_result=True):
        metadata = np.array(metadata)
        points_weights = np.asarray(points_weights)
        
        sorted_idx_lowest = np.argsort(points_weights)
        low_idxs = sorted_idx_lowest[:n_calc]
        high_idxs = sorted_idx_lowest[len(points_weights)-n_calc:]
        
        if len(metadata.shape) < 2:
            metadata_low = metadata[low_idxs]
            metadata_high = metadata[high_idxs]
        else:
            metadata_low = metadata[low_idxs, :]
            metadata_high = metadata[high_idxs, :]
        log("low weigthed samples")
        tpr, fpr = self.calc_dataset_spuriousness(metadata_low, 'low')
        log(f"TPR: {tpr} FPR {fpr}")
        log("high weighted samples")
        tnr, fnr = self.calc_dataset_spuriousness(metadata_high, 'high')
        log(f"TNR: {tnr} FNR {fnr}")
        return tpr, fpr, tnr, fnr
    
    def calculate_dynamic_spuriousness(self, metadata, points_weights):
        percentages = np.arange(0.1,1.0,0.1)
        low_accs = []
        high_accs = []
        for p in percentages:
            n_calc = int(p*metadata.shape[0])
            low_, high_ = self.calculate_spuriousness_fix_n(metadata, points_weights, n_calc, log_result=False)
            low_accs.append(low_)
            high_accs.append(high_)
        log(f"dynamic spuriousness p = {percentages}")
        log(f"Low: {low_accs}")
        log(f"High: {high_accs}")
        return low_accs, high_accs






        
