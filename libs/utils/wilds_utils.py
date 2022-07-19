from wilds import get_dataset
import torch

root_dir = "wilds_data"

dataset_name = "waterbirds"
dataset = get_dataset(dataset=dataset_name, download=False, root_dir=root_dir)

def evaluate_wilds(outputs, labels, metadata):
    results_obj, results_str =dataset.eval(torch.Tensor(outputs), torch.Tensor(labels), torch.Tensor(metadata))
    return results_obj, results_str
