import numpy as np
from libs.model import WeightedCausalClassifier
import os
import shutil
import pandas as pd
from libs.utils.logger import log

def get_data(samples_dict):
    train_baseline = []
    val_baseline = []
    test_baseline = []
    for task in samples_dict:
        y_train = samples_dict[task]['full_features'][:,-1]
        task_data = samples_dict[task]['full_features'][:,:-1]
        
        valdata = samples_dict[task]['valdata']
        testdata = samples_dict[task]['testdata']
        train_baseline.append(task_data)
        
        val_baseline.append(valdata[:,:-1])
        y_val = valdata[:,-1]
        test_baseline.append(testdata[:,:-1])
        y_test = testdata[:,-1]
    if len(train_baseline) > 0:
        train_baseline = np.hstack((train_baseline))
        train_baseline = np.hstack((train_baseline, y_train.reshape(-1,1)))
    if valdata is not None and len(val_baseline) > 0:
        val_baseline = np.hstack((val_baseline))
        y_val = np.array(y_val)
        val_baseline = np.hstack((val_baseline, y_val.reshape(-1,1)))
    if testdata is not None and len(test_baseline) > 0:
        test_baseline = np.hstack((test_baseline))
        y_test = np.array(y_test)
        test_baseline = np.hstack((test_baseline, y_test.reshape(-1,1)))
    return train_baseline, val_baseline, test_baseline

def translate_pca_weight_to_full_weights(feature_map, pca_weights, n_orig_features):
    weights_full = np.zeros((1, n_orig_features-1))
    for node_idx, weight in enumerate(pca_weights):
        try:
            nodes_full = feature_map[node_idx]
            weights_full[0, nodes_full] = weight
        except Exception as e:
            print(e)
    return weights_full.flatten()

def get_features_weights(samples_dict, edge_probs, n_orig_features):
    feature_weights_all = []
    for i, task in enumerate(samples_dict):
        feature_map = samples_dict[task]['pca_feature_mapping']
        features_weights_pca = edge_probs[:, i]
        feature_weights_full = translate_pca_weight_to_full_weights(feature_map, features_weights_pca, n_orig_features)
        feature_weights_all.extend(feature_weights_full.tolist())
    return feature_weights_all

def get_points_weights(traindata, model, feature_weights, epochs=30, lr = 1e-3, l2_penalty=0.1, evaluate_func=None, metadata_val=None,\
    valdata=None, batch_size=64, log_freq=20):
    weighted_clf = WeightedCausalClassifier()
    points_weights = weighted_clf.get_points_weights_mask_once(traindata, model, feature_weights, epochs=epochs, lr=lr, \
                                                               l2_penalty=l2_penalty, evaluate_func=evaluate_func, metadata_val=metadata_val, \
                                                                   valdata=valdata, batch_size=batch_size, log_freq=log_freq)
    # points_weights = [1/p for p in points_w eights.tolist()]
    return points_weights

def store_spurious_images(store_path, files_to_copy):
    if not os.path.isdir(store_path):
        os.makedirs(store_path)
    for file in files_to_copy:
        filename = file.split(os.path.sep)[-1]
        shutil.copy(file, os.path.join(store_path, filename))
    
def group_and_store_images_by_weigts(point_weights, csv_file, train_images_path, metadata_train, n_store=100, store_images=True, \
                                        store_path=None, return_eval_results=True):
    
    sorted_idx_lowest = np.argsort(point_weights)
    n_lowest = sorted_idx_lowest[:n_store]
    n_highest = sorted_idx_lowest[len(point_weights)-n_store:]
    
    metadata_low = metadata_train[n_lowest]
    metadata_high = metadata_train[n_highest]
    
    low_p_spur = len(metadata_low[metadata_low == 0]) / len(metadata_low)
    high_p_spur = len(metadata_high[metadata_high == 0]) / len(metadata_high)
    
    log("% high files from spurious group: {:.3f}".format(high_p_spur))
    log("% low files from spurious group: {:.3f}".format(low_p_spur))
    df = pd.read_csv(csv_file)
    
    if store_images:
        assert store_path is not None
        train_file_paths = df[df['split']==0]['image_path'].tolist()
        train_file_paths = np.asarray(train_file_paths)
        low_files = train_file_paths[n_lowest]
        high_files = train_file_paths[n_highest]
        store_spurious_images(os.path.join(store_path, "low"), low_files)
        store_spurious_images( os.path.join(store_path, "high"), high_files)
    
    if return_eval_results:
        return high_p_spur, low_p_spur, np.abs(high_p_spur-low_p_spur)
        
    
    
