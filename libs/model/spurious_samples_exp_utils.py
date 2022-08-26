import numpy as np
from libs.model import WeightedCausalClassifier

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
    valdata=None, batch_size=64):
    weighted_clf = WeightedCausalClassifier()
    points_weights = weighted_clf.get_points_weights_mask_once(traindata, model, feature_weights, epochs=epochs, lr=lr, \
                                                               l2_penalty=l2_penalty, evaluate_func=evaluate_func, metadata_val=metadata_val, \
                                                                   valdata=valdata, batch_size=batch_size)
    all_points_weights = [1/p for p in points_weights.tolist()]
    return all_points_weights
