import argparse
import os

from libs.core import load_config
from libs.model import *
from libs.model.COmnivore_V import COmnivore_V
from libs.model.LF import LF
from libs.utils import *
from libs.utils.logger import log, set_log_path
from libs.model.spurious_samples_exp_utils import *

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
    if 'feature_path' in args and not isinstance(args.feature_path,type(None)):
        load_path = args.feature_path
    else:
        load_path = dataset_cfg['load_path']
    
    n_orig_features = dataset_cfg['n_orig_features']
    if 'n_pac_features' in dataset_cfg:
        n_pca_features = dataset_cfg['n_pac_features']
    else:
        n_pca_features = None
    global tasks
    tasks = dataset_cfg['tasks']
    fuser = cfg['model']['fuser']
    assert fuser == 'COmnivore_V'

    #########################################################
    # create log folder
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if 'log_path' in args and args.log_path is not None:
        log_path = os.path.join('log','spurious_exp', dataset_name,args.log_path,timestamp)
    else:
        log_path = os.path.join('log','spurious_exp', dataset_name, timestamp)
    
    ensure_path(log_path)
    set_log_path(log_path)


    lf_factory = LF()
    samples_dict = get_samples_dict(load_path,n_orig_features,n_pca_features,tasks)
    metadata_train = np.load(os.path.join(load_path, "metadata_train.npy"))
    metadata_test = np.load(os.path.join(load_path, "metadata_test.npy"))

    metadata_val = np.load(os.path.join(load_path, "metadata_val.npy"))

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
        log_freq = 20
    

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

    log("Training with fused causal estimates...")
    eval_accs_spur = {}
    eval_accs_baselne = {}
    cache_nodes = []
    if 'images_path' in args and not isinstance(args.images_path,type(None)):
        images_path = args.images_path
    else:
        images_path = dataset_cfg['images_path']
    n_save_images = utils_cfg['n_save_images']
    csv_file = os.path.join(images_path, "metadata.csv")
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
        
        best_diff = 0
        best_cb = 0
        for cb in all_negative_balance:
            log(f"###### {cb} ######")
            g_hats, edge_probs = COmnivore.fuse_estimates(cb, n_pca_features, return_probs=True)
            feature_weights = get_features_weights(samples_dict, edge_probs, n_orig_features)
            
            traindata, valdata_processed, testdata_processed, pca_nodes = get_data(samples_dict, g_hats)
            
            if test_duplicate_nodes(pca_nodes, cache_nodes) or len(traindata) == 0:
                print("Nodes cached.. skipping training")
                continue
            points_weights = get_points_weights(traindata, model, feature_weights, epochs, lr, l2, evaluate_func=evaluate_func, \
                                                metadata_val=metadata_val, valdata=valdata_processed, \
                                                batch_size=bs, log_freq=log_freq)
            
            high_p_spur, low_p_spur, diff = group_and_store_images_by_weigts(points_weights, csv_file, images_path, metadata_train, 
                                                                            n_store=n_save_images, store_images=False, store_path=None)
            log(f"% SPURIOUS SAMPLES SEPARATION: {diff}")
            log("TRAIN WITH NO SAMPLE WEIGHTS")
            acc_baseline = train_and_evaluate_end_model(traindata, valdata_processed, metadata_val, testdata_processed, metadata_test, \
                                    generator=rng,
                                    points_weights=None, \
                                    epochs=epochs, \
                                    lr=lr, bs=bs, l2=l2, dropout=dropout, \
                                    model=model, n_layers=n_layers,\
                                    evaluate_func=evaluate_func, log_freq=log_freq, \
                                    tune_by_metric=tune_by_metric, verbose=True)
            log("TRAIN WITH SAMPLE WEIGHT")
            acc_spur = train_and_evaluate_end_model(traindata, valdata_processed, metadata_val, testdata_processed, metadata_test, \
                                    generator=rng, 
                                    points_weights=points_weights, \
                                    epochs=epochs, \
                                    lr=lr, bs=bs, l2=l2, dropout=dropout,\
                                    model=model, n_layers=n_layers,\
                                    evaluate_func=evaluate_func, log_freq=log_freq, \
                                    tune_by_metric=tune_by_metric, verbose=True)
            eval_accs_spur[cb] = acc_spur
            eval_accs_baselne[cb] = acc_baseline
            cache_nodes.append(pca_nodes)
            
            if diff > best_diff:
                best_diff = diff
                best_cb = cb

        best_model_spur, best_cb_eval = get_best_model_acc(eval_accs_spur, tune_by=tune_by_metric, return_best_key=True)
        best_model_base = get_best_model_acc(eval_accs_baselne, tune_by=tune_by_metric)
        
        log(f"BEST SEPARATION: {best_diff} CB: {best_cb}")
        return best_model_spur,best_model_base

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
    parser.add_argument('-img_path', '--images_path', type=str, help='root images path', default=None)
    parser.add_argument('-feat_path', '--feature_path', type=str, help='CLIP features path', default=None)
    args = parser.parse_args()
    spurious_accs, baseline_accs = main(args)
    if baseline_accs is not None:
        print_result(baseline_accs, "baseline")
    if spurious_accs is not None:
        print_result(spurious_accs, "weighted sample")
    os._exit(os.EX_OK)