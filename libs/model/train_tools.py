import os
from libs.model import CausalClassifier, PretrainedCausalClf, MLP, CLIPMLP
from libs.utils import *
from libs.utils.logger import save_graph, log, set_log_path

import numpy as np
import pickle
from tqdm import tqdm

def get_samples_dict(load_path, n_orig_features, n_pca_features=None, tasks=[]):
    samples_dict = {}
    for task in tasks:
        full_features = np.load(os.path.join(load_path, f"{task}_full_train_{n_orig_features}.npy"))
        if task == "irm":
            test_features = np.load(os.path.join(load_path, f"{task}_full_test_{n_orig_features}.npy"))
            val_features = np.load(os.path.join(load_path, f"{task}_full_val_{n_orig_features}.npy"))
        else:
            test_features = np.load(os.path.join(load_path, f"orig_full_test_{n_orig_features}.npy"))
            val_features = np.load(os.path.join(load_path, f"orig_full_val_{n_orig_features}.npy"))
        
        if n_pca_features is not None:
            pca_features = np.load(os.path.join(load_path, f"{task}_pca_train_{n_pca_features}.npy"))
            samples_dict[task] = {
                "full_features": full_features,
                "pca_features": pca_features,
                "G_estimates": [],
                "valdata": val_features,
                "testdata": test_features,
            }
            with open(os.path.join(load_path,f'pca_feature_mapping_{task}_{n_pca_features-1}.pickle'), 'rb') as f:
                pca_mapping = pickle.load(f)
                samples_dict[task]['pca_feature_mapping'] = pca_mapping
        else:
            pca_features = None
            samples_dict[task] = {
                "full_features": full_features,
                "G_estimates": []
            }
        
    return samples_dict

def run_notears_lfs(samples_dict,tasks,lf_func, lf_name, use_cpdag=False, log_graph = True):
    G_estimates = {}
    for task in tqdm(tasks):
        pca_features = samples_dict[task]['pca_features']
        dag, W, cpdag = lf_func(pca_features)
        if not use_cpdag:
            G_estimates[task] = dag
        else:
            G_estimates[task] = cpdag
        if log_graph:
            save_graph(G_estimates[task], title=f"{task} LF {lf_name}")
    return G_estimates

def run_classic_lfs(samples_dict,tasks, lf_func, lf_name, transpose=True, pycausal=False, log_graph = True):
    G_estimates = {}
    for task in tqdm(tasks):
        pca_features = samples_dict[task]['pca_features']
        if not pycausal:
            dag = lf_func(pca_features)
        else:
            dag = lf_func(pca_features, lf_name)
        if not transpose:
            G_estimates[task] = dag
        else:
            G_estimates[task] = dag.T
        if log_graph:
            save_graph(G_estimates[task], title=f"{task} LF {lf_name}")
    return G_estimates

def translate_pca_to_full(feature_map, pca_nodes):
    nodes_full = []
    for node in pca_nodes:
        try:
            nodes_full.extend(feature_map[node])
        except Exception as e:
            print(feature_map)
    return nodes_full

def get_data_from_feat_label_array(samples_dict, G_estimates=None, scale=False):
    train_baseline = []
    val_baseline = []
    y_val = []
    test_baseline = []
    y_test = []
    pca_nodes = {}
    all_train_nodes = {}
    if G_estimates is None:
        for task in tqdm(samples_dict):
            y_train = samples_dict[task]['full_features'][:,-1]
            task_data = samples_dict[task]['full_features'][:,:-1]
            
            valdata = samples_dict[task]['valdata']
            testdata = samples_dict[task]['testdata']
            train_baseline.append(task_data)
            
            val_baseline.append(valdata[:,:-1])
            y_val = valdata[:,-1]
            test_baseline.append(testdata[:,:-1])
            y_test = testdata[:,-1]
            nodes_to_train = list(np.arange(0, task_data.shape[1]))
            all_train_nodes[task] = nodes_to_train
    else:
        for task_lf in G_estimates:
            feature_map = samples_dict[task_lf]['pca_feature_mapping']
            y_train = samples_dict[task_lf]['full_features'][:,-1]
            task_data = samples_dict[task_lf]['full_features'][:,:-1]
            # if task_lf == 'irm':
            valdata = samples_dict[task_lf]['valdata']
            testdata = samples_dict[task_lf]['testdata']
            G = G_estimates[task_lf]
            causal_clf = CausalClassifier(G)
            selected_pca_nodes = causal_clf.nodes_to_train
            pca_nodes[task_lf] = selected_pca_nodes
            nodes_to_train = translate_pca_to_full(feature_map, selected_pca_nodes)
            all_train_nodes[task_lf] = nodes_to_train
            log(F"N NODES TO TRAIN {len(nodes_to_train)}")
            if len(causal_clf.nodes_to_train) > 0:
                train_baseline.append(np.take(task_data, nodes_to_train, axis=1))
                
                val_baseline.append(np.take(valdata[:,:-1], nodes_to_train, axis=1))
                y_val = valdata[:,-1]
            
                test_baseline.append(np.take(testdata[:,:-1], nodes_to_train, axis=1))
                y_test = testdata[:,-1]
            else:
                continue
    if len(train_baseline) > 0:
        train_baseline = np.hstack((train_baseline))
        if scale:
            train_baseline = scale_data(train_baseline)
        train_baseline = np.hstack((train_baseline, y_train.reshape(-1,1)))
    if valdata is not None and len(val_baseline) > 0:
        val_baseline = np.hstack((val_baseline))
        if scale:
            val_baseline = scale_data(val_baseline)
        y_val = np.array(y_val)
        val_baseline = np.hstack((val_baseline, y_val.reshape(-1,1)))
    if testdata is not None and len(test_baseline) > 0:
        test_baseline = np.hstack((test_baseline))
        if scale:
            test_baseline = scale_data(test_baseline)
        y_test = np.array(y_test)
        test_baseline = np.hstack((test_baseline, y_test.reshape(-1,1)))
    return train_baseline, val_baseline, test_baseline, pca_nodes, all_train_nodes

def scale_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def train_and_evaluate_end_model(traindata, valdata, metadata_val, testdata, metadata_test, generator, \
                                    epochs=20, lr=1e-3, bs=32, l2=0.1, dropout=0.1, model=CLIPMLP, n_layers=2, evaluate_func=None, log_freq=None, \
                                        tune_by_metric='acc_wg'):
    baseline_accs = {}

    if len(traindata) == 0:
        return 0

    baseline = CausalClassifier()
    baseline.train_baseline(model, traindata, batch_size=bs, lr=lr, epochs=epochs,\
                            verbose=False, valdata=valdata, metadata_val=metadata_val, \
                                n_layers=n_layers, l2=l2, dropout=dropout, generator=generator, \
                            evaluate_func=evaluate_func,log_freq=log_freq,tune_by_metric=tune_by_metric)

    outputs_val, labels_val, _ = baseline.evaluate(baseline.best_chkpt, valdata)
    results_obj_val, results_str_val = evaluate_func(outputs_val, labels_val, metadata_val)
    log(f"Val \n {results_str_val}")

    outputs_test, labels_test, _ = baseline.evaluate(baseline.model, testdata)

    results_obj_test, results_str_test = evaluate_func(outputs_test, labels_test, metadata_test)

    log(f"Test \n {results_str_test}")
    baseline_accs['val'] = {k:v for k,v in results_obj_val.items()}
    baseline_accs['test'] = {k:v for k,v in results_obj_test.items()}
    return baseline_accs

def log_config(lr, l2, bs, dropout, n_layers):
    log(f"END MODEL HYPERPARAMS: lr = {lr} | l2 = {l2} | bs = {bs} | dropout = {dropout} | n_layers = {n_layers}")

def test_baseline_nodes(pca_nodes, n_pca_features):
    matches = []
    for key in pca_nodes:
        nodes_ = pca_nodes[key]
        if len(nodes_) == n_pca_features-1:
            match = True
        else:
            match = False
        matches.append(match)
    if np.array(matches).all() == True:
        return True
    return False

def pretrained_model_inference(dataset_name, model_path=None, features=[], nodes_to_train=[], n_feats_orig=0, \
    metadata=[], evaluate_func=None):
    model = PretrainedCausalClf(dataset_name, model_path)
    preds, labels = model.infer(features, nodes_to_train, n_feats_orig)
    results_obj_test, results_str_test = evaluate_func(preds, labels, metadata)
    log(f"{results_str_test}")
    # accs['test'] = {k:v for k,v in results_obj_test.items()}
    return results_obj_test
    
def test_duplicate_nodes(pca_nodes, cache_nodes):
    if len(cache_nodes) == 0:
        return False
    for i, cache in enumerate(cache_nodes):
        matches = []
        for key in cache:
            c_nodes = cache[key]
            c_nodes = np.sort(c_nodes)
            p_nodes = pca_nodes[key]
            p_nodes = np.sort(p_nodes)
            if np.array_equal(c_nodes, p_nodes) == True:
                match = True
            else:
                match = False
            matches.append(match)
        if np.array(matches).all() == True:
            return True
    return False
    
def get_best_model_acc(eval_accs, tune_by='acc_wg'):
    best_val_acc = float('-inf')
    best_key = None
    for key in eval_accs:
        if eval_accs[key]['val'][tune_by] > best_val_acc:
            best_val_acc = eval_accs[key]['val'][tune_by]
            best_key = key
    return eval_accs[best_key]