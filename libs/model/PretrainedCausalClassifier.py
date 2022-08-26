from torch import nn
from .CausalClassifier import CausalClassifier
import torch
import torchvision
import numpy as np

class Resnet50_Extractor(nn.Module):
    def __init__(self, model):
        super(Resnet50_Extractor, self).__init__()
        self.classifier = nn.Sequential(list(model.children())[-1])
    
    def forward(self, x):
        clf = self.classifier(x)
        return clf
    
class PretrainedCausalClf(CausalClassifier):
    def __init__(self, dataset_name, model_path):
        super(CausalClassifier, self).__init__()
        self.model = self.initialize_torchvision_model(
                name='resnet50',
                d_out=2,
                **{})
        self.model.eval()
        model_path = f"/hdd2/dyah/wilds/logs/{dataset_name}_seed:0_epoch:best_model.pth"
        state_dict = torch.load(model_path)['algorithm']
        state_dict = self.modify_keys(state_dict)
        self.model.load_state_dict(state_dict)
        self.model = Resnet50_Extractor(self.model)
        self.model.cuda()
    
    def modify_keys(self, state_dict):
        new_state_dict = {}
        for old_key in state_dict:
            new_key = old_key[6:]
            new_state_dict[new_key] = state_dict[old_key]
        return new_state_dict
    
    def add_zeros_to_features(self, features, nodes_to_train, n_feats_orig):
        filled_features = np.zeros((features.shape[0], n_feats_orig))
        for i, node_id in enumerate(nodes_to_train):
            filled_features[:, node_id] = features[:, i]
        filled_features[:, -1] = features[:, -1]
        return filled_features
    
    def infer(self, features, nodes_to_train, n_feats_orig):
        if features.shape[1] < n_feats_orig:
            features = self.add_zeros_to_features(features, nodes_to_train, n_feats_orig)
        preds, labels, _ = self.evaluate(self.model, features, nodes_to_train=[i for i in range(features.shape[1]-1)], batch_size=64)
        return preds, labels
    
    def initialize_torchvision_model(self, name, d_out, **kwargs):
        # get constructor and last layer names
        if name == 'wideresnet50':
            constructor_name = 'wide_resnet50_2'
            last_layer_name = 'fc'
        elif name == 'densenet121':
            constructor_name = name
            last_layer_name = 'classifier'
        elif name in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
            constructor_name = name
            last_layer_name = 'fc'
        else:
            raise ValueError(f'Torchvision model {name} not recognized')
        # construct the default model, which has the default last layer
        constructor = getattr(torchvision.models, constructor_name)
        model = constructor(**kwargs)
        # adjust the last layer
        d_features = getattr(model, last_layer_name).in_features
        if d_out is None:  # want to initialize a featurizer model
            last_layer = Identity(d_features)
            model.d_out = d_features
        else: # want to initialize a classifier for a particular num_classes
            last_layer = nn.Linear(d_features, d_out)
            model.d_out = d_out
        setattr(model, last_layer_name, last_layer)

        return model
