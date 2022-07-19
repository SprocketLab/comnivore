import numpy as np
from utils.metrics import shd, sid
from utils.graph_modules import modify_single_edge, get_ordered_edge_sets
import networkx as nx

from tqdm import tqdm
class State:
    def __init__(self, current_G, current_cost):
        self.G = current_G
        self.cost = current_cost

class HillClimbing:
    def __init__(self, G_lambdas, initial_state=None, maxiter=1000, dist_metric = sid, weights = []):
        self.G_lambdas = G_lambdas
        self.maxiter = maxiter
        self.dist_metric = dist_metric

        if initial_state is None:
            n_node = self.G_lambdas[0].shape[0]
            initial_state = nx.to_numpy_matrix(nx.gnp_random_graph(n_node, 0.5, directed=True))
        self.state = State(initial_state, self.objective(initial_state))

        self.edge_set = get_ordered_edge_sets(self.state.G)
        self.smallest_cost = 1
        if len(weights) == 0:
            self.weights = [1 for i in range(len(self.G_lambdas))]
        else:
            self.weights = weights
    
    def objective(self, G_candidate):
        sum_dist = 0
        for i, G in enumerate(self.G_lambdas):
            sum_dist += self.weights[i] * self.dist_metric(G_candidate, G)
        return sum_dist/len(self.G_lambdas)
    
    def get_neighbors(self):
        neighbors = []
        for edge in self.edge_set:
            G = modify_single_edge(self.state.G, edge)
            neighbors.append(G)
        return neighbors
    
    def evaluate_candidates(self, candidates):
        best_candidates = {}
        for candidate in candidates:
            cost = self.objective(candidate)
            if cost <= self.state.cost:
                if cost not in best_candidates:
                    best_candidates[cost] = [candidate]
                else:
                    best_candidates[cost].append(candidate)
        if len(list(best_candidates.keys())) == 0:
            return None
        best_cost = sorted(list(best_candidates.keys()), reverse=True)
        best_neighbors = best_candidates[best_cost[0]]
        neighbors_idx = np.linspace(0,len(best_neighbors), len(best_neighbors), dtype=int, endpoint=False)
        return best_neighbors[np.random.choice(neighbors_idx)]


    def optimize(self):
        print("starting optimizer...")
        print("initial objective value:", self.state.cost)

        for i in range(0, self.maxiter):
            possible_candidates = self.get_neighbors()
            better_candidate = self.evaluate_candidates(possible_candidates)
            if better_candidate is None:
                print("no more better candidates")
                break
            else:
                self.state = State(better_candidate, self.objective(better_candidate))
            print(i, self.state.cost)
        
            
        
class SimulatedAnnealing:
    def __init__(self, G_lambdas, initial_state='avg', maxiter=1000, decrease_t_iter=100, dist_metric = shd, T=10000, alpha=0.1, reduction_rule='geometric', gt_dag=None, weights=[]):
        self.G_lambdas = G_lambdas
        self.maxiter = maxiter
        self.decrease_t_iter = decrease_t_iter
        self.T = T
        self.alpha = alpha
        self.dist_metric = dist_metric
        print("ANNEALING")
        if len(weights) == 0:
            self.weights = [1 for i in range(len(self.G_lambdas))]
        else:
            self.weights = weights
        if initial_state == 'random':
            n_node = self.G_lambdas[0].shape[0]
            self.initial_G = nx.to_numpy_matrix(nx.gnp_random_graph(n_node, 0.5, directed=True))
        elif initial_state == 'avg':
            self.initial_G = self.get_average_G()
        else:
            assert type(initial_state) != str
            self.initial_G = np.copy(initial_state)
        self.state = State(self.initial_G, self.objective(self.initial_G))

        self.edge_set = get_ordered_edge_sets(self.state.G)
        self.reduction_rule = reduction_rule
        self.smallest_cost = 1

        if gt_dag is not None:
            self.G_star_cost = self.objective(gt_dag)
            print('G* objective is', self.G_star_cost)

    def get_average_G(self):
        distances = []
        for i, G_i in enumerate(self.G_lambdas):
            distances.append(self.calculate_avg_dist(G_i, i))
        best_idx = np.argmin(distances)
        best_average_G = self.G_lambdas[best_idx]
        return best_average_G
    
    def calculate_avg_dist(self, Gi, G_idx):
        sum_dist = 0
        for i, G in enumerate(self.G_lambdas):
            if i == G_idx:
                continue
            sum_dist += self.dist_metric(Gi, G)
        return sum_dist/len(self.G_lambdas)

    def objective(self, G_candidate):
        sum_dist = 0
        for i, G in enumerate(self.G_lambdas):
            sum_dist += self.weights[i] * self.dist_metric(G_candidate, G)
        return sum_dist/len(self.G_lambdas)

    def get_neighbors(self):
        neighbors = []
        for edge in self.edge_set:
            G = modify_single_edge(self.state.G, edge)
            neighbors.append(G)
        return neighbors
    
    def decrease_T(self):
        if self.reduction_rule == 'geometric':
            self.T = self.T*self.alpha
        elif self.reduction_rule == 'linear':
            self.T = self.T-self.alpha
        else:
            self.T = self.T / (1+ self.T*self.alpha)
    
    def calculate_p_accept(self, delta_cost):
        return np.exp(-delta_cost/self.T)

    def random_move(self):
        neighbors = self.get_neighbors()
        neighbors_idx = np.linspace(0,len(neighbors), len(neighbors), dtype=int, endpoint=False)
        return neighbors[np.random.choice(neighbors_idx)]
    
    def random_move_optimized(self):
        radom_edge_idx = np.random.choice([i for i in range(len(self.edge_set))])
        G = modify_single_edge(self.state.G, self.edge_set[radom_edge_idx])
        return G



    def optimize(self, verbose=True):
        print("initial objective value:", self.state.cost)
        # smallest_cost_attained = False
        for i in range(0, self.maxiter, self.decrease_t_iter):
            for j in range(self.decrease_t_iter):
                candidate = self.random_move_optimized()
                possible_state = State(candidate, self.objective(candidate))
                if possible_state.cost < self.state.cost:
                    self.state = possible_state
                else:
                    delta_cost = possible_state.cost - self.state.cost
                    p_accept = self.calculate_p_accept(delta_cost)
                    self.state = np.random.choice([self.state, possible_state], p=[1-p_accept, p_accept])
            if verbose and i%10 == 0:
                print(i, self.state.cost)
            self.decrease_T()
        
        # print("optimizer finished")
        print("final objective value", self.state.cost)
        return self.state


