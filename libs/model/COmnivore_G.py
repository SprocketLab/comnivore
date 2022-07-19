import numpy as np
from ..utils.metrics import shd
from .Best_G_Estimator import SimulatedAnnealing
from tqdm import tqdm

class COmnivore_G:
    def __init__(self, G_estimates, n_triplets=7, min_iters = 100, max_iters = 10000, step = 100):
        self.G_estimates = G_estimates
        self.lf_names = list(self.G_estimates.keys())
        self.n_triplets = n_triplets
        self.tasks = list(self.G_estimates[self.lf_names[0]].keys())
        self.min_iters = min_iters
        self.max_iters = max_iters
        self.step = step
        # self.n_iters = np.array([i for i in range(min_iters, max_iters+step, step)])
    
    def get_distance_matrix(self):
        n_features = self.G_estimates[self.lf_names[0]][self.tasks[0]].shape[1]
        L_matrix = np.zeros((len(self.tasks), len(self.lf_names), n_features, n_features))      

        for i, task in enumerate(self.tasks):
            for j, lf in enumerate(list(self.G_estimates.keys())):
                lf_g = self.G_estimates[lf][task]
                L_matrix[i, j, :, :] = lf_g
        
        # compute rate of LF agreement => average dist between pairs of LFs across all tasks
        dist_mat = np.zeros((L_matrix.shape[1], L_matrix.shape[1]))
        for LF_id_1 in range(L_matrix.shape[1]):
            LF_curr = L_matrix[:,LF_id_1, :, :]
            for LF_id_2 in range(L_matrix.shape[1]):
                if LF_id_1 == LF_id_2:
                    continue
                LF_other = L_matrix[:,LF_id_2, :, :]
                dist_tasks = []
                for task in range(LF_curr.shape[0]):
                    dist_tasks.append(shd(LF_curr[task], LF_other[task]))
                dist_mat[LF_id_1][LF_id_2] = np.mean(dist_tasks)
        self.distance_matrix = dist_mat
        return dist_mat
    
    def triplet_solver(self, dist_a_b, dist_a_c, dist_b_c):
        return (dist_a_b + dist_a_c - dist_b_c) / 2

    def get_empirical_LF_acc(self):
        LF_empirical_acc = []
        distance_matrix = self.get_distance_matrix()
        LF_all = np.arange(0, distance_matrix.shape[0], dtype=int)
        for LF_idx in LF_all:
            lf_other = np.delete(LF_all, LF_idx)
            LF_accs = []
            for i in range(self.n_triplets):
                lf_rand_0, lf_rand_1 = np.random.choice(lf_other, 2, replace=False)
                dist_a_b = distance_matrix[LF_idx, lf_rand_0]
                dist_a_c = distance_matrix[LF_idx, lf_rand_1]
                dist_b_c = distance_matrix[lf_rand_0, lf_rand_1]
                LF_accs.append(self.triplet_solver(dist_a_b, dist_a_c, dist_b_c))
            emp_acc = 1/np.median(LF_accs) if np.median(LF_accs) != 0 else 0
            LF_empirical_acc.append(emp_acc)
        
        self.empirical_acc = LF_empirical_acc
        return LF_empirical_acc

    def get_task_lfs(self, task):
        task_lfs = []
        for lf in self.G_estimates:
            task_lfs.append(self.G_estimates[lf][task])
        return task_lfs
    
    def fuse_estimates(self):
        empirical_acc = self.get_empirical_LF_acc()
        g_hats_per_task = {}
        for n, task in tqdm(enumerate(self.tasks)):
            lfs = self.get_task_lfs(task)
            initial_state = 'random'
            g_hats_per_task[task] = []
            for i, iteration in tqdm(enumerate(range(self.min_iters, self.max_iters+self.step, self.step))):
                if i > 0: 
                    initial_state = np.copy(best_estimated_G)
                search_optimizer = SimulatedAnnealing(lfs, initial_state, self.step, reduction_rule='geometric', weights=empirical_acc)
                search_optimizer.optimize(verbose=False)
                best_estimated_G = search_optimizer.state.G
                g_hats_per_task[task].append(best_estimated_G)
        return g_hats_per_task
                        