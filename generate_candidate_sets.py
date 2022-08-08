from libs.core import load_config
from libs.model import Extractor_CLIP, Extractor_VAE, Extractor_CNN, Phi
from libs.candidate_sets import Candidate_Set
from libs.utils.logger import log, set_log_path
from libs.utils import *

import argparse
import os
import pickle
from datetime import datetime

import numpy as np

from libs.datasets import DATASETS, WILDS_DATASETS, DOMAINBED_DATASETS

def get_model_dict():
    return {
        'CLIP': Extractor_CLIP,
        'VAE': Extractor_VAE,
        'CNN': Extractor_CNN,
    }
    
def extract_features(extractor_model, dataloader):
    phi = Phi(extractor_model)
    features = phi.get_z_features(dataloader)
    return features

def store_features(store_path, features, mode, type_):
    if "components" in type_ or "metadata" in type_:
        np.save(os.path.join(store_path,f"{type_}_{mode}.npy"), features)
    else:
        np.save(os.path.join(store_path,f"{type_}_{mode}_{features.shape[1]}.npy"), features)

def store_mapping(store_path, z_hidden, mapping, task=None):
    filepath = os.path.join(store_path, f'pca_feature_mapping_{task}_{z_hidden}.pickle')
    with open(filepath, 'wb') as handle:
        pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def main(args):
    cfg = load_config(args.config)
    dataset_cfg = cfg['data']['dataset']
    dataset_name = dataset_cfg['dataset_name']
    root_path = dataset_cfg['root_path']
    feature_path = dataset_cfg['feature_path']
    
    store_dir = os.path.join(root_path, dataset_name, feature_path)
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if 'log_path' in args and args.log_path is not None:
        log_path = os.path.join('log',args.log_path, dataset_name, "EXTRACTION LOG",timestamp)
    else:
        log_path = os.path.join('log', dataset_name, "EXTRACTION LOG", timestamp)
    ensure_path(log_path)
    set_log_path(log_path)
    
    extraction_config = cfg['extraction_pipeline']
    extraction_bs = extraction_config['extraction_batch_size']
    extraction_reshape_size = extraction_config['reshape_size']
    if 'z_hidden' in cfg:
        z_hidden = cfg['z_hidden']
    else:
        z_hidden = extraction_config['z_hidden']
    
    environments = dataset_cfg['tasks']
    candidate_set = Candidate_Set(dataset_name, extraction_reshape_size, extraction_bs)
    
    train_loaders = candidate_set.get_all_train_loader_by_tasks(environments)
    test_loader = candidate_set.get_test_loader()
    val_loader = candidate_set.get_val_loader()
    
    train_metadata = candidate_set.get_train_metadata()
    test_metadata = candidate_set.get_test_metadata()
    val_metadata = candidate_set.get_val_metadata()
    
    store_features(store_dir, train_metadata, "train", "metadata")
    store_features(store_dir, test_metadata, "test", "metadata")
    store_features(store_dir, val_metadata, "val", "metadata")
    
    model = extraction_config['extractor_model']
    extractor_model = get_model_dict()[model](z_hidden)
    
    for i, train_loader in enumerate(train_loaders):
        env = environments[i]
        log(f"Extracting {env} training features....")
        features_full, features_pca, feature_mapping, components = extract_features(extractor_model, train_loader)
        store_mapping(store_dir, z_hidden, feature_mapping, environments[i])
        store_features(store_dir, features_full, "train", f"{env}_full")
        store_features(store_dir, features_pca, "train", f"{env}_pca")
        store_features(store_dir, components, "train", f"{env}_components")
    
    if test_loader is not None:
        log("Extracting test features...")
        test_features_full, tes_features_pca, _, _ = extract_features(extractor_model, test_loader)
        store_features(store_dir, test_features_full, "test", "orig_full")
        store_features(store_dir, tes_features_pca, "test", "orig_pca")
   
    log("Extracting val features...")
    val_features_full, val_features_pca, _, _ = extract_features(extractor_model, val_loader)
    
    store_features(store_dir, val_features_full, "val", "orig_full")
    store_features(store_dir, val_features_pca, "val", "orig_pca")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config file path', required=True)
    parser.add_argument('-z', '--z_hidden', type=int, help='hidden dimension for dimensionality reduction',)
    args = parser.parse_args()
    main(args)
    os._exit(os.EX_OK)
    