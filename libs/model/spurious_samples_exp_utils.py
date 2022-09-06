import numpy as np
from libs.model import WeightedCausalClassifier, CausalClassifier, MLP, CLIPMLP
import os
import shutil
import pandas as pd
from libs.utils.logger import log


def get_data(samples_dict, G_estimates):
    train_baseline = []
    val_baseline = []
    test_baseline = []
    pca_nodes = {}
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

        G = G_estimates[task]
        causal_clf = CausalClassifier(G)
        selected_pca_nodes = causal_clf.nodes_to_train
        pca_nodes[task] = selected_pca_nodes

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
    return train_baseline, val_baseline, test_baseline, pca_nodes

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

def get_base_predictor(traindata, model, \
                        feature_weights, epochs=30, \
                        lr = 1e-3, l2_penalty=0.1, evaluate_func=None, 
                        metadata_val=None,\
                        valdata=None, batch_size=64, \
                        log_freq=20):
    weighted_clf = WeightedCausalClassifier()
    base_predictor, f_x_base = weighted_clf.get_base_predictor(traindata, model, feature_weights, \
                                                                epochs=epochs, lr=lr, \
                                                               l2_penalty=l2_penalty, \
                                                               evaluate_func=evaluate_func, \
                                                               metadata_val=metadata_val, \
                                                                valdata=valdata, batch_size=batch_size, \
                                                                log_freq=log_freq)
    return base_predictor, f_x_base
                                    
def get_points_weights_with_base_predictor(base_predictor, f_x, train_data, feature_weights, batch_size):
    weighted_clf = WeightedCausalClassifier()
    points_weights = weighted_clf.get_points_weights_with_base_predictor(base_predictor, f_x, train_data, feature_weights, batch_size)
    if points_weights is not None:
        points_weights = [1/p for p in points_weights.tolist()] ### try functions that make stuffs low twhen the weight are high
        return points_weights
    else:
        return None #None is when all predicted edge weights are too small -- no causal features predicted

def get_points_weights(traindata, model, \
                        feature_weights, epochs=30, \
                        lr = 1e-3, l2_penalty=0.1, evaluate_func=None, 
                        metadata_val=None,\
                        valdata=None, batch_size=64, \
                        log_freq=20, zero_one = False, \
                        p_zero = None, tune_by='acc_wg',
                        non_causal=False):

    weighted_clf = WeightedCausalClassifier()
    points_weights, masked_feats_idxs = weighted_clf.get_points_weights_mask_once(traindata, model, feature_weights, \
                                                                epochs=epochs, lr=lr, \
                                                                l2_penalty=l2_penalty, evaluate_func=evaluate_func, \
                                                                metadata_val=metadata_val, \
                                                                valdata=valdata, batch_size=batch_size, \
                                                                log_freq=log_freq, tune_by=tune_by, \
                                                                non_causal=non_causal)
    if points_weights is not None:
        points_weights = [1/p for p in points_weights.tolist()]
        # points_weights = np.log(points_weights)
        if zero_one:
            assert p_zero is not None and p_zero > 0
            points_weights = binarize_score(points_weights, p_zero)
        return points_weights, weighted_clf, masked_feats_idxs
    else:
        return None #None is when all predicted edge weights are too small -- no causal features predicted

def sigmoid(p):
    p = np.asarray(p)
    return 1/(1 + np.exp(-p))

def binarize_score(points_weights, p_zero):
    # sort, get index of p_zero% lowest, 0 if in that group
    points_weights = np.asarray(points_weights)
    sorted_idx = np.argsort(points_weights)
    n_lowest_idx = sorted_idx[:int(p_zero * len(sorted_idx))]
    points_weights[n_lowest_idx] = 0
    points_weights[points_weights > 0] = 1
    return points_weights

def store_spurious_images(store_path, files_to_copy, scores=None):
    if not os.path.isdir(store_path):
        os.makedirs(store_path)
    for i, file in enumerate(files_to_copy):
        filename = file.split(os.path.sep)[-1]
        if scores is not None:
            score = round(scores[i], 3)
            filename = f"{str(score)}_{filename}"
        shutil.copy(file, os.path.join(store_path, filename))

def evaluate_trained_model(trained_model, traindata, valdata, metadata_val, testdata, metadata_test, generator, \
                            bs=32, evaluate_func=None,):
    accs_ = {}
    clf = WeightedCausalClassifier()
    outputs_val, labels_val, _ = clf.evaluate(trained_model.best_chkpt, valdata, batch_size=bs)
    results_obj_val, results_str_val = evaluate_func(outputs_val, labels_val, metadata_val)
    log(f"Val \n {results_str_val}")
    outputs_test, labels_test, _ = clf.evaluate(trained_model.best_chkpt, testdata, batch_size=bs)

    results_obj_test, results_str_test = evaluate_func(outputs_test, labels_test, metadata_test)

    log(f"Test \n {results_str_test}")
    accs_['val'] = {k:v for k,v in results_obj_val.items()}
    accs_['test'] = {k:v for k,v in results_obj_test.items()}
    return accs_

def train_and_evaluate_end_model_weighted(traindata, valdata, metadata_val, testdata, metadata_test, generator, points_weights=[], \
                                epochs=20, lr=1e-3, bs=32, l2=0.1, dropout=0.1, model=CLIPMLP, n_layers=2, \
                                evaluate_func=None, log_freq=20, \
                                tune_by_metric='acc_wg', verbose=True):
    accs_ = {}
    if len(traindata) == 0:
        return 0
    clf = WeightedCausalClassifier()
    clf.train_end_model(model, traindata, evaluate_func, points_weights, \
                        valdata=valdata, metadata_val=metadata_val,\
                        batch_size=bs, lr=lr, epochs=epochs, l2_weight=l2, \
                        verbose=verbose, log_freq=log_freq, \
                        tune_by_metric=tune_by_metric)

    outputs_val, labels_val, _ = clf.evaluate(clf.best_chkpt, valdata)
    results_obj_val, results_str_val = evaluate_func(outputs_val, labels_val, metadata_val)
    log(f"Val \n {results_str_val}")
    outputs_test, labels_test, _ = clf.evaluate(clf.best_chkpt, testdata)

    results_obj_test, results_str_test = evaluate_func(outputs_test, labels_test, metadata_test)

    log(f"Test \n {results_str_test}")
    accs_['val'] = {k:v for k,v in results_obj_val.items()}
    accs_['test'] = {k:v for k,v in results_obj_test.items()}
    return accs_

    
def group_and_store_images_by_weigts(point_weights, csv_file, metadata_train, n_store=100, store_images=True, \
                                        store_path=None, return_eval_results=True, root_dir = None):
    
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
        train_file_paths = df[df['split']==0]['img_filename'].tolist()
        if root_dir is not None:
            train_file_paths = [os.path.join(root_dir, f) for f in train_file_paths]
        train_file_paths = np.asarray(train_file_paths)
        low_files = train_file_paths[n_lowest]
        high_files = train_file_paths[n_highest]
        store_spurious_images(os.path.join(store_path, "low"), low_files, np.asarray(point_weights)[n_lowest])
        store_spurious_images( os.path.join(store_path, "high"), high_files, np.asarray(point_weights)[n_highest])
    
    if return_eval_results:
        return high_p_spur, low_p_spur, np.abs(high_p_spur-low_p_spur)
        
    
def analyze_weights(point_weights):
    point_weights = np.array(point_weights)
    log(f"Max: {np.amax(point_weights)} | Min: {np.amin(point_weights)} | Mean: {np.mean(point_weights)} | Median: {np.median(point_weights)}")