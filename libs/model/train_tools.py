import argparse
import os
import sys


from libs.core import load_config
from libs.model import *
from libs.model.COmnivore_V import COmnivore_V
from libs.model.LF import LF
from libs.utils import *
from libs.utils.wilds_utils import evaluate_wilds
from libs.utils.logger import log_graph, log, set_log_path
from libs.utils.metrics import shd

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import networkx as nx
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
import networkx as nx
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def get_samples_dict(load_path,n_orig_features,n_pca_features,tasks):
    samples_dict = {}
    for task in tasks:
        full_features = np.load(os.path.join(load_path, f"{task}_full_train_{n_orig_features}.npy"))
        pca_features = np.load(os.path.join(load_path, f"{task}_pca_train_{n_pca_features}.npy"))
        #pca_features[:, :-1] = pca_features[:, :-1] - np.mean(pca_features[:, :-1], axis=0)
        samples_dict[task] = {
            "full_features": full_features,
            "pca_features": pca_features,
            "G_estimates": []
        }
        with open(os.path.join(load_path,f'pca_feature_mapping_{task}_{n_pca_features-1}.pickle'), 'rb') as f:
            pca_mapping = pickle.load(f)
            samples_dict[task]['pca_feature_mapping'] = pca_mapping
    return samples_dict

def run_notears_lfs(samples_dict,tasks,lf_func, lf_name, use_cpdag=False):
    G_estimates = {}
    for task in tqdm(tasks):
        pca_features = samples_dict[task]['pca_features']
        dag, W, cpdag = lf_func(pca_features)
        if not use_cpdag:
            G_estimates[task] = dag
        else:
            G_estimates[task] = cpdag
        log_graph(G_estimates[task], title=f"{task} LF {lf_name}")
    return G_estimates

def run_classic_lfs(samples_dict,tasks, lf_func, lf_name, transpose=True, pycausal=False):
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
        log_graph(G_estimates[task], title=f"{task} LF {lf_name}")
    return G_estimates

def translate_pca_to_full(feature_map, pca_nodes):
    nodes_full = []
    for node in pca_nodes:
        try:
            nodes_full.extend(feature_map[node])
        except Exception as e:
            print(feature_map)
    return nodes_full

def get_data_from_feat_label_array(samples_dict, valdata=None, testdata=None, G_estimates=None, scale=False):
    train_baseline = []
    val_baseline = []
    y_val = []
    test_baseline = []
    y_test = []
    if G_estimates is None:
        for task in tqdm(samples_dict):
            y_train = samples_dict[task]['full_features'][:,-1]
            task_data = samples_dict[task]['full_features'][:,:-1]
            # if G_estimates is None:
            train_baseline.append(task_data)
            if valdata is not None:
                val_baseline.append(valdata[:,:-1])
                y_val.append(valdata[:,-1])
            if testdata is not None:
                test_baseline.append(testdata[:,:-1])
                y_test.append(testdata[:,-1])
    else:
        for task_lf in G_estimates:
            feature_map = samples_dict[task_lf]['pca_feature_mapping']
            y_train = samples_dict[task_lf]['full_features'][:,-1]
            task_data = samples_dict[task_lf]['full_features'][:,:-1]
            G = G_estimates[task_lf]
            causal_clf = CausalClassifier(G)
            nodes_to_train = translate_pca_to_full(feature_map, causal_clf.nodes_to_train)
            print("N NODES TO TRAIN", len(nodes_to_train))
            if len(causal_clf.nodes_to_train) > 0:
                train_baseline.append(np.take(task_data, nodes_to_train, axis=1))
                if valdata is not None:
                    val_baseline.append(np.take(valdata, nodes_to_train, axis=1))
                    y_val.append(valdata[:,-1])
                if testdata is not None:
                    test_baseline.append(np.take(testdata, nodes_to_train, axis=1))
                    y_test.append(testdata[:,-1])
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
    return train_baseline, val_baseline, test_baseline

def scale_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def train_and_evaluate_end_model(samples_dict, valdata, metadata_val, testdata, metadata_test, generator, epochs=20, lr=1e-3, bs=32, l2=0, model=MLP, G_estimates=None, scale=False):
    baseline_accs = {}
    traindata, valdata, testdata = get_data_from_feat_label_array(samples_dict, valdata, testdata, G_estimates, scale)

    if len(traindata) == 0:
        return 0

    baseline = CausalClassifier()
    baseline.train_baseline(model, traindata, epochs=epochs, lr=lr, verbose=False, 
                            batch_size=bs, valdata=valdata, metadata_val=metadata_val, l2=l2,generator=generator)

    outputs_val, labels_val, _ = baseline.evaluate(baseline.best_chkpt, valdata)
    results_obj_val, results_str_val = evaluate_wilds(torch.Tensor(outputs_val), torch.Tensor(labels_val), torch.Tensor(metadata_val))
    log(f"Val \n {results_str_val}")

    outputs_test, labels_test, _ = baseline.evaluate(baseline.model, testdata)
    results_obj_test, results_str_test = evaluate_wilds(torch.Tensor(outputs_test), torch.Tensor(labels_test), torch.Tensor(metadata_test))

    log(f"Test \n {results_str_test}")
    baseline_accs['val'] = {k:v for k,v in results_obj_val.items()}
    baseline_accs['test'] = {k:v for k,v in results_obj_test.items()}
    return baseline_accs

def log_config(lr, l2, bs):
    log("END MODEL HYPERPARAMS")
    log(f"lr = {lr} | l2 = {l2} | bs = {bs}")