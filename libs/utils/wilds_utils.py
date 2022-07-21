from wilds import get_dataset
import torch

root_dir = "wilds_data"
# dataset_name = "waterbirds"

class WILDS_utils:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = get_dataset(dataset=dataset_name, download=False, root_dir=root_dir)
    
    def evaluate_wilds(self, outputs, labels, metadata):
        results_obj, results_str = self.dataset.eval(torch.Tensor(outputs), torch.Tensor(labels), torch.Tensor(metadata))
        return results_obj, results_str
    