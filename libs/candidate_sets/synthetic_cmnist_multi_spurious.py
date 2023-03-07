from .synthetic_coloredmnist_candidate import Synthetic_ColoredMNIST_Candidate_Set
import numpy as np

class Multi_Spurious_CMNIST_Candidate_Set(Synthetic_ColoredMNIST_Candidate_Set):
    def __init__(self, batch_size, root_cmnist_path):
        super(Multi_Spurious_CMNIST_Candidate_Set, self).__init__(batch_size, root_cmnist_path)
    
    def get_test_metadata(self):
        test_rows = self.df.loc[self.df["split"] == 1]
        random_metadata = np.array(test_rows['random'].tolist()).reshape(-1,1)
        spurious_feats_n_metadata = np.array(test_rows['spurious_feats_n'].tolist()).reshape(-1,1)
        metadata = np.hstack((random_metadata, spurious_feats_n_metadata))
        return metadata
    
    def get_val_metadata(self):
        val_rows = self.df.loc[self.df["split"] == 2]
        random_metadata = np.array(val_rows['random'].tolist()).reshape(-1,1)
        spurious_feats_n_metadata = np.array(val_rows['spurious_feats_n'].tolist()).reshape(-1,1)
        metadata = np.hstack((random_metadata, spurious_feats_n_metadata))
        # print('HERE', metadata)
        # exit()
        return metadata
    
    def get_train_metadata(self):
        train_rows = self.df.loc[self.df["split"] == 0]
        random_metadata = np.array(train_rows['random'].tolist()).reshape(-1,1)
        spurious_feats_n_metadata = np.array(train_rows['spurious_feats_n'].tolist()).reshape(-1,1)
        metadata = np.hstack((random_metadata, spurious_feats_n_metadata))
        return metadata
