import numpy as np
import networkx as nx
from snorkel.labeling.model import LabelModel

class COmnivore_V:
    def __init__(self, G_estimates, snorkel_lr=1e-3, snorkel_ep=100):
        self.G_estimates = G_estimates
        self.T = 1
        self.lf_names = list(self.G_estimates.keys())
        self.tasks = list(self.G_estimates[self.lf_names[0]].keys())
        self.snorkel_lr = snorkel_lr
        self.snorkel_ep = snorkel_ep
    
    def get_g_hat_from_edge_preds(self, edge_preds):
        # edge_preds: n_features x n_task
        # return 1 graph for each task
        n_features, n_task = edge_preds.shape
        g_hats = {}
        for task in range(n_task):
            g_task = np.zeros((n_features+1, n_features+1))
            edges = edge_preds[:,task]
            edge_idx = np.argwhere(edges==1).flatten()
            g_task[edge_idx,-1]=1
            g_hats[self.tasks[task]] = g_task
        return g_hats
    
    def get_cb(self, cb):
        return np.array([cb] + [0 for i in range(2 ** self.T - 2)] + [1 - cb])
    
    # predict edge for each node to label (e.g., predict whether G is causal (has path) or not to label)
    # will run FS prediction n_node times and get prediction for each node
    def get_ws_edge_prediction(self, cb, n_pca_features):
        class_balance = self.get_cb(cb)
        m = len(self.lf_names)
        v = 1
        label_node = n_pca_features -1
        n_features = n_pca_features - 1
        edge_predictions = []
        edge_probs = []
        n_tasks = len(self.tasks)
        for feat_idx in range(n_features):
            node = feat_idx
            L_edge = np.zeros((n_tasks, m))
            for lf_idx, lf_name in enumerate(self.lf_names):
                for task_idx, task in enumerate(self.tasks):
                    lf_g_hat = self.G_estimates[lf_name][task]
                    causal_path = nx.has_path(nx.DiGraph(lf_g_hat),node,label_node)
                    anti_causal_path = nx.has_path(nx.DiGraph(lf_g_hat),label_node,node)
                    if causal_path and not anti_causal_path:
                        L_edge[task_idx][lf_idx] = 1.
                    elif anti_causal_path:
                        L_edge[task_idx][lf_idx] = -1.
                    else:
                        L_edge[task_idx][lf_idx] = 0
            triplet_model = LabelModel(
                cardinality=2, verbose=False, 
            )
            triplet_model.fit(
                L_edge,
                n_epochs=self.snorkel_ep, seed=123, lr=self.snorkel_lr,
                class_balance=class_balance, 
                progress_bar=False
            )
            preds = triplet_model.predict(L_edge)
            proba = triplet_model.predict_proba(L_edge)
            edge_predictions.append(preds.flatten())
            edge_probs.append(proba[:,1].flatten())
        edge_predictions = np.vstack(edge_predictions) # n_features x n_task
        edge_probs = np.vstack(edge_probs)
        g_hats = self.get_g_hat_from_edge_preds(edge_predictions)
        return g_hats, edge_probs

    def fuse_estimates(self, cb, n_pca_features, return_probs=False):
        g_hats_edge, edge_probs = self.get_ws_edge_prediction(cb, n_pca_features)
        if not return_probs:
            return g_hats_edge
        else:
            return g_hats_edge, edge_probs
