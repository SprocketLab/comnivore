import argparse
import os

from libs.core import load_config
from libs.model import *
from libs.model.COmnivore_V import COmnivore_V
from libs.model.COmnivore_G import COmnivore_G
from libs.model.LF import LF
from libs.utils import *
from libs.utils.logger import log, set_log_path

import numpy as np
from datetime import datetime
import torch

from libs.utils.wilds_utils import WILDS_utils
from libs.utils.generic_utils import Generic_utils
from libs.datasets import WILDS_DATASETS, DOMAINBED_DATASETS, SYNTHETIC_DATASETS

cuda = True if torch.cuda.is_available() else False
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def main(args):
    # may be we need to implement load from checkpoint function here

    #load config
    cfg = load_config(args.config)
    rng = fix_random_seed(cfg.get('seed', 2022))

    # load tasks params
    dataset_cfg = cfg['data']['dataset']
    dataset_name = dataset_cfg['dataset_name']
    load_path = dataset_cfg['load_path']
    
    n_orig_features = dataset_cfg['n_orig_features']
    if 'n_pac_features' in dataset_cfg:
        n_pca_features = dataset_cfg['n_pac_features']
    else:
        n_pca_features = None
    global tasks
    tasks = dataset_cfg['tasks']
    fuser = cfg['model']['fuser']


    #########################################################
    # create log folder
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if 'log_path' in args and args.log_path is not None:
        log_path = os.path.join('log', args.log_path, dataset_name, fuser,timestamp)
    else:
        log_path = os.path.join('log', dataset_name, fuser, timestamp)
    ensure_path(log_path)
    set_log_path(log_path)


    lf_factory = LF()
    samples_dict = get_samples_dict(load_path,n_orig_features,n_pca_features,tasks)

    metadata_test = np.load(os.path.join(load_path, "metadata_test.npy"))
    # testdata = np.load(os.path.join(load_path, f"orig_full_test_{n_orig_features}.npy"))

    metadata_val = np.load(os.path.join(load_path, "metadata_val.npy"))
    # valdata = np.load(os.path.join(load_path, f"orig_full_val_{n_orig_features}.npy"))

    ##################################################################################
    # set up optimizer
    opt = cfg['opt']
    epochs = opt['epochs']
    if 'learning_rate' in args and not isinstance(args.learning_rate,type(None)):
        lr = args.learning_rate
    else:
        lr = opt['lr']
    if 'l2_regularizer' in args and not isinstance(args.l2_regularizer,type(None)):
        l2 = args.l2_regularizer
    else:
        l2 = opt['l2']
    if 'batch_size' in args and not isinstance(args.batch_size,type(None)):
        bs = args.batch_size
    else:
        bs = cfg['data']['batch_size']


    model_cfg = cfg['model']['output_model']
    model = select_model(model_cfg)

    ##################################################################################
    # load params for pipline
    pipline = cfg['pipeline']
    
    evaluate_func = None
    if dataset_name in WILDS_DATASETS:
        evaluate_func = WILDS_utils(dataset_name).evaluate_wilds
    elif dataset_name in DOMAINBED_DATASETS or dataset_name in SYNTHETIC_DATASETS:
        evaluate_func = Generic_utils().evaluate
    
    if 'dropout' in opt:
        dropout = opt['dropout']
    else:
        dropout = 0.0
    if 'tune_by' in cfg['model']:
        tune_by_metric = cfg['model']['tune_by']
    else:
        tune_by_metric = 'acc_wg'
    if 'n_layers' in opt:
        n_layers = opt['n_layers']
    else:
        n_layers = 2
    log_config(lr, l2, bs, dropout, n_layers)
    baseline_accs = None
    if 'utils' in cfg:
        utils_cfg = cfg['utils']
        log_freq = utils_cfg['log_freq']
    else:
        log_freq = 50
    
    if 'train' in cfg:
        train = cfg['train']
    else:
        train = True
    if pipline['baseline']:
        traindata, valdata_processed, testdata_processed, _, nodes_to_train = get_data_from_feat_label_array(samples_dict, G_estimates=None, scale=False)
        if train:
            log("Training baseline....")
            baseline_accs = train_and_evaluate_end_model(traindata, valdata_processed, metadata_val, testdata_processed, metadata_test,rng, \
                                                        epochs, lr, bs, l2, dropout=dropout, model=model, n_layers=n_layers, \
                                                            evaluate_func=evaluate_func, log_freq=log_freq, \
                                                            tune_by_metric=tune_by_metric)
        else:
            log("Baseline inference...")
            baseline_accs = {}
            accs_val = pretrained_model_inference(dataset_name, model_path=None, features=valdata_processed, nodes_to_train=nodes_to_train[tasks[0]], n_feats_orig=n_orig_features, \
                metadata=metadata_val, evaluate_func=evaluate_func)
            accs_test = pretrained_model_inference(dataset_name, model_path=None, features=testdata_processed, nodes_to_train=nodes_to_train[tasks[0]], n_feats_orig=n_orig_features, \
                metadata=metadata_test, evaluate_func=evaluate_func)
            baseline_accs['val'] = accs_val
            baseline_accs['test'] = accs_test
    if pipline['fused_causal'] == False:
        return baseline_accs, None
    
    active_lfs = cfg['model']['active_lfs']
    G_estimates = {}
    for lf in active_lfs['notears']:
        log(f"Running {lf}...")
        G_estimates[lf] = run_notears_lfs(samples_dict,tasks, lf_factory.lf_dict[lf], lf, False)

    for lf in active_lfs['classic']:
        log(f"Running {lf}...")
        G_estimates[lf] = run_classic_lfs(samples_dict,tasks, lf_factory.lf_dict[lf], lf, False)

    for lf in active_lfs['pycausal']:
        log(f"Running {lf}...")
        G_estimates[lf] = run_classic_lfs(samples_dict, tasks,lf_factory.lf_dict['pycausal'], lf, False, pycausal=True)

    if pipline['indiv_training']:
        log("Training using individual LF estimates...")
        for lf in G_estimates:
            log(lf)
            traindata, valdata_processed, testdata_processed, _, _ = get_data_from_feat_label_array(samples_dict, G_estimates=G_estimates[lf], scale=False)
            train_and_evaluate_end_model(traindata, valdata_processed, metadata_val, testdata_processed, metadata_test,rng, \
                                        epochs, lr, bs, l2, dropout=dropout, model=model, n_layers=n_layers,\
                                            G_estimates=G_estimates[lf], evaluate_func=evaluate_func, \
                                             log_freq=log_freq, tune_by_metric=tune_by_metric)

    log("Training with fused causal estimates...")

    #################################################################################
    # load params for COmnivore
    #################################################################################

    
    log(f"FUSE ALGORITHM: {fuser}")
    eval_accs_all = {}
    cache_nodes = []
    if fuser == 'COmnivore_V':
        COmnivore_params = opt['comnivore_v']
        all_negative_balance = np.arange(COmnivore_params['all_negative_balance'][0],COmnivore_params['all_negative_balance'][1],COmnivore_params['all_negative_balance'][2])
        if 'snorkel_lr' in args and args.snorkel_lr is not None:
            snorkel_lr = args.snorkel_lr
        else:
            snorkel_lr = COmnivore_params['snorkel_lr']
        if 'snorkel_epochs' in args and args.snorkel_epochs is not None:
            snorkel_ep = args.snorkel_epochs
        else:
            snorkel_ep = COmnivore_params['snorkel_ep']
        log(f"SNORKEL PARAMS: lr {snorkel_lr} | ep {snorkel_ep}")
        COmnivore = COmnivore_V(G_estimates, snorkel_lr, snorkel_ep)
        
        for cb in all_negative_balance:
            log(f"###### {cb} ######")
            g_hats = COmnivore.fuse_estimates(cb, n_pca_features)
            traindata, valdata_processed, testdata_processed, pca_nodes, all_train_nodes = get_data_from_feat_label_array(samples_dict, G_estimates=g_hats, scale=False)
            train_nodes = [all_train_nodes[key] for key in all_train_nodes]
            if test_baseline_nodes(pca_nodes, n_pca_features):
                print("Same as baseline nodes.. skipping training")
                continue
            if train:
                if not test_duplicate_nodes(pca_nodes, cache_nodes) and len(traindata) > 0:
                    eval_accs = train_and_evaluate_end_model(traindata, valdata_processed, metadata_val, testdata_processed, metadata_test,rng, \
                                                epochs, lr, bs, l2, dropout=dropout, model=model, n_layers=n_layers, \
                                                    evaluate_func=evaluate_func, \
                                                    log_freq=log_freq, tune_by_metric=tune_by_metric)
                    eval_accs_all[cb] = eval_accs
                    cache_nodes.append(pca_nodes)
                else:
                    print("Nodes cached.. skipping training")
            else:
                if len(traindata) > 0:
                    eval_accs_all[cb] = {}
                    accs_val = pretrained_model_inference(dataset_name, model_path=None, features=valdata_processed, nodes_to_train=all_train_nodes[tasks[0]], n_feats_orig=n_orig_features, \
                        metadata=metadata_val, evaluate_func=evaluate_func)
                    accs_test = pretrained_model_inference(dataset_name, model_path=None, features=testdata_processed, nodes_to_train=all_train_nodes[tasks[0]], n_feats_orig=n_orig_features, \
                        metadata=metadata_test, evaluate_func=evaluate_func)
                    eval_accs_all[cb]['val'] = accs_val
                    eval_accs_all[cb]['test'] = accs_test
    elif fuser == 'COmnivore_G':
        COmnivore_params = opt['comnivore_g']
        n_triplets = COmnivore_params['n_triplets']
        min_iters = COmnivore_params['min_iters']
        max_iters = COmnivore_params['max_iters']
        step = COmnivore_params['step']
        COmnivore = COmnivore_G(G_estimates, n_triplets, min_iters, max_iters, step)
        g_hats_per_task = COmnivore.fuse_estimates()
        n_iters = np.array([i for i in range(min_iters, max_iters+step, step)])
        for i, iter_ in enumerate(n_iters):
            log(f"##### ITER: {iter_} #####")
            g_hats = {}
            for task in g_hats_per_task:
                g_hats[task] = g_hats_per_task[task][i]
            traindata, valdata_processed, testdata_processed, pca_nodes, all_train_nodes = get_data_from_feat_label_array(samples_dict, G_estimates=g_hats, scale=False)
            if test_baseline_nodes(pca_nodes, n_pca_features):
                print("Same as baseline nodes.. skipping training")
                continue
            if not test_duplicate_nodes(pca_nodes, cache_nodes) and len(traindata) > 0:
                eval_accs = train_and_evaluate_end_model(traindata, valdata_processed, metadata_val, testdata_processed, metadata_test,rng, \
                                epochs, lr, bs, l2, dropout=dropout, model=model, n_layers=n_layers, \
                                    G_estimates=g_hats, evaluate_func=evaluate_func, \
                                    log_freq=log_freq, tune_by_metric=tune_by_metric)
                eval_accs_all[iter_] = eval_accs
                cache_nodes.append(pca_nodes)
            else:
                print("Nodes cached, skipping training on these nodes")
    best_model_eval = get_best_model_acc(eval_accs_all, tune_by=tune_by_metric)
        
    return baseline_accs, best_model_eval

    #################################################################################

def print_result(result_obj, mode="baseline"):
    if 'val' in result_obj:
        log(f"\n{mode} val")
        for key in list(result_obj['val'].keys()):
            val_acc = result_obj['val'][key]
            if val_acc <= 1.0:
                log("{}: {:.3f}".format(key, val_acc))
    if 'test' in result_obj:
        log(f"\n{mode} test")
        for key in list(result_obj['test'].keys()):
            test_acc = result_obj['test'][key]
            if test_acc <= 1.0:
                log("{}: {:.3f}".format(key, test_acc))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config file path', required=True)
    parser.add_argument('-lr', '--learning_rate', type=float, help='end model training learning rate')
    parser.add_argument('-l2', '--l2_regularizer', type=float, help='end model l2 regularizer')
    parser.add_argument('-bs', '--batch_size', type=int, help='end model training batch size')
    parser.add_argument('-s_lr', '--snorkel_lr', type=float, help='snorkel learning rate')
    parser.add_argument('-s_ep', '--snorkel_epochs', type=int, help='snorkel epochs')
    parser.add_argument('-log', '--log_path', type=str, help='log path', default=None)
    args = parser.parse_args()
    baseline_accs, best_model_eval = main(args)
    if baseline_accs is not None:
        print_result(baseline_accs, "baseline")
    if best_model_eval is not None:
        print_result(best_model_eval, "ours")
    os._exit(os.EX_OK)