import numpy as np
from libs.utils.logger import log

class Spuriousness_Profiler:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    
    def calc_synthethic_cmnist_spuriousness(self, metadata):
        p_spur = metadata[metadata == 0].shape[0] / metadata.shape[0]
        return p_spur
    
    def calc_waterbirds_spuriousness(self, metadata):
        spur_idxs = np.argwhere(metadata[:, 0] == metadata[:, 1])
        p_spur = spur_idxs.shape[0] / metadata.shape[0]
        return p_spur
    
    def calc_celebA_spuriousness(self, metadata):
        # y (metadata index 1) = 1  : Blonde | cofounder (metadata index 0) = 1 : Male
        # spur_idxs = np.argwhere((metadata[:, 1] ==  1) & (metadata[:, 0] == 0))
        spur_idxs = np.argwhere((metadata[:, 1] ==  0) & (metadata[:, 0] == 1))
        # high_spur_idxs = np.argwhere(((metadata_high[:, 1] ==  1) & (metadata_high[:, 0] == 0)) | ((metadata_high[:, 1] ==  0) & (metadata_high[:, 0] == 1)))
        p_spur = spur_idxs.shape[0] / metadata.shape[0]
        return p_spur

    def calc_dataset_spuriousness(self, metadata):
        if self.dataset_name == 'Synthetic_ColoredMNIST':
            return self.calc_synthethic_cmnist_spuriousness(metadata)
        elif self.dataset_name == 'waterbirds':
            return self.calc_waterbirds_spuriousness(metadata)
        elif self.dataset_name == 'celebA':
            return self.calc_celebA_spuriousness(metadata)

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
        
        low_ = self.calc_dataset_spuriousness(metadata_low)
        high_ = self.calc_dataset_spuriousness(metadata_high)
        if log_result:
            log("% high files from spurious group: {:.3f}".format(high_))
            log("% low files from spurious group: {:.3f}".format(low_))
        return low_, high_
    
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






        
