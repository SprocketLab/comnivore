import argparse
import os

from libs.core import load_config
from libs.model import *
from libs.model.COmnivore_V import COmnivore_V
from libs.model.LF import LF
from libs.utils import *
from libs.utils.logger import log, set_log_path
from libs.model.spurious_samples_exp_utils import *
from libs.model.train_tools import train_and_evaluate_end_model as train_and_evaluate_end_model_causal

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
        log_path = os.path.join('log','SPURIOUS_NEW', dataset_name, args.log_path,timestamp)
    else:
        log_path = os.path.join('log','SPURIOUS_NEW', dataset_name, timestamp)
    
    ensure_path(log_path)
    set_log_path(log_path)

    lf_factory = LF()
    samples_dict = get_samples_dict(load_path,n_orig_features,n_pca_features,tasks)
    
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
    if 'dropout' in args and not isinstance(args.dropout,type(None)):
        dropout = args.dropout
    else:
        dropout = opt['dropout']

    if 'tune_by' in cfg['model']:
        tune_by_metric = cfg['model']['tune_by']
    else:
        tune_by_metric = 'acc_wg'
    if 'n_layers' in opt:
        n_layers = opt['n_layers']
    else:
        n_layers = 2

    log_config(lr, l2, bs, dropout)
    
    if 'utils' in cfg:
        utils_cfg = cfg['utils']
        log_freq = utils_cfg['log_freq']
    else:
        log_freq = 20

    zero_one = False
    p_zero = None
    if 'weighting_scheme' in cfg:
        weighting_cfg = cfg['weighting_scheme']
        if 'zero_one' in weighting_cfg:
            zero_one = weighting_cfg['zero_one']
        if 'p_zero' in args and not isinstance(args.p_zero,type(None)):
            p_zero = args.p_zero
        elif 'p_zero' in weighting_cfg:
            p_zero = weighting_cfg['p_zero']
            # log(f"P_ZERO {p_zero}")

    model_cfg = cfg['model']['output_model']
    model = select_model(model_cfg)
    
    evaluate_func = None
    if dataset_name in WILDS_DATASETS:
        evaluate_func = WILDS_utils(dataset_name).evaluate_wilds
    elif dataset_name in DOMAINBED_DATASETS or dataset_name in SYNTHETIC_DATASETS:
        evaluate_func = Generic_utils().evaluate
    

    active_lfs = cfg['model']['active_lfs']
    G_estimates = {}
    for lf in active_lfs['notears']:
        log(f"Running {lf}...")
        G_estimates[lf] = run_notears_lfs(samples_dict,tasks, lf_factory.lf_dict[lf], lf, False, log_graph=False)

    for lf in active_lfs['classic']:
        log(f"Running {lf}...")
        G_estimates[lf] = run_classic_lfs(samples_dict,tasks, lf_factory.lf_dict[lf], lf, False, log_graph=False)

    for lf in active_lfs['pycausal']:
        log(f"Running {lf}...")
        G_estimates[lf] = run_classic_lfs(samples_dict, tasks,lf_factory.lf_dict['pycausal'], lf, False, pycausal=True, log_graph=False)

    log("Training with fused causal estimates...")
    eval_accs_spur = {}
    eval_accs_baselne = {}
    eval_accs_remove_feats = {}
    eval_accs_combined = {}
    
    cache_nodes = []
    if 'images_path' in args and not isinstance(args.images_path,type(None)):
        images_path = args.images_path
        csv_file = os.path.join(images_path, "metadata.csv")
    else:
        if 'images_path' in dataset_cfg:
            images_path = dataset_cfg['images_path']
            if 'metadata_file_name' not in dataset_cfg:
                csv_file = os.path.join(images_path, "metadata.csv")
            else:
                metadata_file_name = dataset_cfg['metadata_file_name']
                csv_file = os.path.join(images_path, metadata_file_name)
        else:
            image_path = None
            csv_file = None
    if 'n_save_images' in utils_cfg:
        n_save_images = utils_cfg['n_save_images']
    else:
        n_save_images = 20
    
    pipeline = cfg['pipeline']
    if 'remove_features' in pipeline:
        remove_features = pipeline['remove_features']
    else:
        remove_features = False
    if 'sample_weighting' in pipeline:
        sample_weighting = pipeline['sample_weighting']
    else:
        sample_weighting = True

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
            
        COmnivore = COmnivore_V(G_estimates, snorkel_lr, snorkel_ep)
        
        best_diff = float("-inf")
        best_cb = 0
        for cb in all_negative_balance:
            log(f"###### {cb} ######")
            g_hats, edge_probs = COmnivore.fuse_estimates(cb, n_pca_features, return_probs=True)
            feature_weights = get_features_weights(samples_dict, edge_probs, n_orig_features)
            traindata, valdata_processed, testdata_processed, pca_nodes = get_data(samples_dict, g_hats)
            
            if test_duplicate_nodes(pca_nodes, cache_nodes) or len(traindata) == 0:
                print("Nodes cached.. skipping training")
                continue
            if test_empty_nodes(pca_nodes):
                print("No causal features predicted, skipping training")
                continue
            # print("CAUSAL FEATURES TO COMPUTE WEIGHT")
            points_weights, base_predictor, _ = get_points_weights(traindata, model, \
                                                feature_weights, epochs, lr, \
                                                l2, evaluate_func=evaluate_func, \
                                                metadata_val=metadata_val, valdata=valdata_processed, \
                                                batch_size=bs, log_freq=log_freq, zero_one=zero_one, \
                                                p_zero=p_zero, tune_by=tune_by_metric)
            # points_weights = np.random.rand(traindata.shape[0])
            if points_weights is None:
                print("No causal features predicted, skipping training")
                continue
            
            if sample_weighting:
                log("="*100)
                log("BASELINE")
                
                acc_baseline = evaluate_trained_model(base_predictor, 
                                traindata, valdata=valdata_processed, \
                                metadata_val=metadata_val, \
                                testdata=testdata_processed, \
                                metadata_test=metadata_test, generator=rng, \
                                bs=bs, evaluate_func=evaluate_func,)
                analyze_weights(points_weights)
                
                if csv_file is not None:
                    metadata_train = np.load(os.path.join(load_path, "metadata_train.npy"))
                    high_p_spur, low_p_spur, diff = group_and_store_images_by_weigts(points_weights, csv_file, \
                                                                                    metadata_train, \
                                                                                    n_store=n_save_images, store_images=True, 
                                                                                    store_path=os.path.join('spurious_samples_exp',f'{dataset_name}'),
                                                                                    root_dir = os.path.join(images_path),\
                                                                                    dataset_name = dataset_name)
                    log("% SPURIOUS SAMPLES SEPARATION: {:.3f}".format(diff))
                    if diff > best_diff:
                        best_diff = diff
                        best_cb = cb
                
                log("="*100)
                log("WITH SAMPLE WEIGHT")
                acc_spur = train_and_evaluate_end_model_weighted(traindata, valdata_processed, \
                                        metadata_val, testdata_processed, \
                                        metadata_test, \
                                        generator=rng, 
                                        points_weights=points_weights, \
                                        epochs=epochs, \
                                        lr=lr, bs=bs, l2=l2, dropout=dropout,\
                                        model=model, n_layers=n_layers,\
                                        evaluate_func=evaluate_func, log_freq=log_freq, \
                                        tune_by_metric=tune_by_metric, verbose=True)
                eval_accs_spur[cb] = acc_spur
                eval_accs_baselne[cb] = acc_baseline
                
            if remove_features:
                train_causal, val_causal, test_causal, _, _ = get_data_from_feat_label_array(samples_dict, G_estimates=g_hats, scale=False)
                log("="*100)
                log("REMOVE CAUSAL ONLY")
                if not test_baseline_nodes(pca_nodes, n_pca_features):
                    acc_remove = train_and_evaluate_end_model_causal(train_causal, val_causal, metadata_val, \
                                        test_causal, metadata_test,rng, \
                                        epochs, lr, bs, l2, dropout=dropout, \
                                        model=model, n_layers=n_layers, \
                                        evaluate_func=evaluate_func, \
                                        log_freq=log_freq, tune_by_metric=tune_by_metric)
                    eval_accs_remove_feats[cb] = acc_remove
                log("="*100)
                log("REMOVE CAUSAL + WEIGHTED")
                acc_combined = train_and_evaluate_end_model_weighted(train_causal, val_causal,\
                                    metadata_val, test_causal, \
                                    metadata_test, \
                                    generator=rng, 
                                    points_weights=points_weights, \
                                    epochs=epochs, \
                                    lr=lr, bs=bs, l2=l2, dropout=dropout,\
                                    model=model, n_layers=n_layers,\
                                    evaluate_func=evaluate_func, log_freq=log_freq, \
                                    tune_by_metric=tune_by_metric, verbose=True)
                
                eval_accs_combined[cb] = acc_combined
            cache_nodes.append(pca_nodes)
        
        best_model_base = {}
        best_model_spur = {}
        best_model_remove = {}
        best_model_combined = {}
        
        if sample_weighting:
            best_model_base, best_cb_ = get_best_model_acc(eval_accs_baselne, tune_by=tune_by_metric, return_best_key=True)
            log(f"BEST CB BASELINE: {best_cb_}")
            best_model_spur, best_cb_ = get_best_model_acc(eval_accs_spur, tune_by=tune_by_metric, return_best_key=True)
            log(f"BEST CB WEIGHTING: {best_cb_}")
        if remove_features:
            best_model_remove, best_cb_ = get_best_model_acc(eval_accs_remove_feats, tune_by=tune_by_metric, return_best_key=True)
            log(f"BEST CB REMOVE FEATS: {best_cb_}")
            best_model_combined, best_cb_ = get_best_model_acc(eval_accs_combined, tune_by=tune_by_metric, return_best_key=True)
            log(f"BEST CB COMBINED: {best_cb_}")
        if csv_file is not None:
            log(f"BEST SEPARATION: {best_diff} CB: {best_cb}")
        
        return best_model_spur, best_model_base, best_model_remove, best_model_combined

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
    parser.add_argument('-do', '--dropout', type=float, help='dropout')
    parser.add_argument('-log', '--log_path', type=str, help='log path', default=None)
    parser.add_argument('-img_path', '--images_path', type=str, help='root images path', default=None)
    parser.add_argument('-feat_path', '--feature_path', type=str, help='CLIP features path', default=None)
    parser.add_argument('-p_zero', '--p_zero', type=float, help='% of sample weight set to 0 if using 0-1 weighting scheme', default=None)
    args = parser.parse_args()
    spurious_accs, baseline_accs, accs_remove, accs_combined = main(args)
    if len(baseline_accs) > 0:
        print_result(baseline_accs, "baseline")
    if len(spurious_accs) > 0:
        print_result(spurious_accs, "weighted sample")
    if len(accs_remove) > 0:
        print_result(accs_remove, "remove non-causal features only")
    if len(accs_combined) > 0:
        print_result(accs_combined, "combined remove + weighting")
    os._exit(os.EX_OK)