import numpy as np
from tqdm import tqdm

def greedy_plot(all_nodes, train_fn, traindata, testdata, verbose=False):
    #greedyly find best combinations for the plot
    acc_orig = []
    acc_interv = []
    best_queue = []
    nodes = list(np.copy(all_nodes))
    n_features = 0
    while len(best_queue) < len(all_nodes):
        orig_ = []
        interv_ = []
        feature_order = []
        n_features += 1
        if verbose:
            print(f"#### N FEATURES {n_features} ####")
        for i, f in enumerate(nodes):
            best_queue.append(f)
            feature_order.append(f)
            result_obj = train_fn(best_queue)
            if result_obj is not None:
                orig_.append(result_obj['acc_avg'])
                interv_.append(result_obj['acc_wg'])
                best_queue.pop()
        best_acc_idx = np.argmax(np.array(interv_))
        acc_orig.append(orig_[best_acc_idx])
        acc_interv.append(interv_[best_acc_idx])

        best_queue.append(feature_order[best_acc_idx])
        nodes.remove(feature_order[best_acc_idx])
        print(f"best queue: {len(best_queue)}")
    return best_queue, acc_orig, acc_interv

def test_indiv_features(all_nodes, train_fn, traindata, testdata):
    acc_dict = {}
    for node in all_nodes:
        print('NODE', node)
        results_obj = train_fn([node], traindata=traindata, testdata=testdata)
        acc_dict[node] = results_obj
    return acc_dict

