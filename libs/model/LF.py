from numpy.core.fromnumeric import take
#from libs.notears.nonlinear import linear, nonlinear
import libs.notears.nonlinear as nonlinear
import libs.notears.linear as linear
import numpy as np

import pandas as pd
from pycausal.pycausal import pycausal
from pycausal import search as s

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.CDNOD import cdnod


from causallearn.utils.cit import fisherz, chisq, kci, gsq, mv_fisherz
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search

# from causallearn.search.FCMBased.lingam import CAMUV
# from causallearn.search.FCMBased import GIN
# from causallearn.search.Granger.Granger import Granger

import lingam

from cdt.causality.graph import GS, MMPC, IAMB, Inter_IAMB
from cdt.metrics import retrieve_adjacency_matrix

# from cdt.causality.pairwise import RECI
# from cdt.causality.graph import 
# from cdt.causality.graph import SAM

from causallearn.search.FCMBased import lingam

class LF:
    def __init__(self):
        self.lf_dict = {
            'NoTears Sobolev': self.LF_nonlinear_sobolev,
            'NoTears MLP': self.LF_nonlinear_mlp,
            'PC': self.LF_pc,
            'FCI': self.LF_fci,
            'Exact Search': self.LF_bic_exact_search,
            'Lingam': self.LF_lingam,
            'pycausal': self.LF_pycausal,
            'MMPC': self.LF_MMPC,
            'GS': self.LF_GS,
            'IAMB': self.LF_IAMB,
            'Inter_IAMB': self.LF_Inter_IAMB,
            'ICA_Lingam': self.LF_ICA_Lingam,
            'Direct_Lingam': self.LF_Direct_Lingam, 
            'Var_Lingam': self.LF_Var_Lingam,
            'RCD': self.LF_RCD,
        }
    
    def get_operating_subgraph(features, dag_manual):
        operating_nodes = np.argwhere(dag_manual[:, -1] ==1).flatten().tolist()
        operating_nodes.append(features.shape[1]-1)
        subgraph = features[:, operating_nodes]
        return subgraph

    def get_adjacency(self, W):
        dag = np.copy(W)
        dag[dag > 0] = 1
        dag[dag < 0] = -1
        return dag
    
    def process_cpdag(self, cpdag):
        cpdag_T = np.copy(cpdag.T)
        cpdag_T[cpdag_T > 0] = 0
        cpdag_T[cpdag_T < 0] = -1
        cpdag_T = -1 * cpdag_T
        cpdag_processed = cpdag_T + np.abs(cpdag)
        return cpdag_processed

    def LF_manual_color_feature(self, feature, bw_feature, tolerance=1e-2, topn=6):
        assert bw_feature.all() != None
        dist = np.abs(feature[:, :-1] - bw_feature[:, :-1])
        # print(np.amax(dist), np.amin(dist))
        zero_indexes = np.argwhere(dist <= tolerance)
        color_features = np.unique(zero_indexes[:, 1])
        # n_zero_dict = {}
        # for i, f in enumerate(color_features):
        #     # if np.std(feature[:, f]) < epsilon:
        #     #     continue
        #     n_zero = np.argwhere(dist[:,f]<=tolerance).flatten().shape[0]
        #     # if n_zero > (0.9) * feature.shape[0]:
        #     #     continue
        #     n_zero_dict[f] = n_zero

        # n_zero_sorted = [k for k, _ in sorted(n_zero_dict.items(), key=lambda item: item[1], reverse=True)]
        dag = np.zeros((feature.shape[1], feature.shape[1]))
        if topn:
            dag[color_features[:topn],-1] = 1
        else:
            dag[color_features, -1] = 1
        return dag, color_features

    def LF_linear(self, feature):
        '''
        Linear DAGs with No Tears: https://arxiv.org/pdf/1803.01422.pdf
        https://github.com/xunzheng/notears
        '''
        W_linear = linear.notears_linear(feature, lambda1=0.1, loss_type='l2')
        dag = self.get_adjacency(W_linear)
        processed_cpdag = self.process_cpdag(dag)
        # if np.argwhere(dag_linear < 0).shape[0] > 0:
        #     self.cpdag_to_dags(dag_linear)
        return dag, W_linear, processed_cpdag
    
    def LF_nonlinear_sobolev(self, feature):
        '''
        Sobolev Nonlinear DAGs with No Tears: https://arxiv.org/pdf/1909.13189.pdf
        https://github.com/xunzheng/notears
        '''
        d = feature.shape[1]
        model = nonlinear.NotearsSobolev(d, k=1)
        W_basis_exp = nonlinear.notears_nonlinear(model, feature.astype(np.float32), lambda1=0.01, lambda2=0.01)
        dag = self.get_adjacency(W_basis_exp)
        processed_cpdag = self.process_cpdag(dag)
        return dag, W_basis_exp, processed_cpdag
    
    def LF_nonlinear_mlp(self, feature):
        '''
        MLP Nonlinear DAGs with No Tears: https://arxiv.org/pdf/1909.13189.pdf
        https://github.com/xunzheng/notears
        '''
        d = feature.shape[1]
        model = nonlinear.NotearsMLP(dims=[d, 10, 1], bias=True)
        W_mlp = nonlinear.notears_nonlinear(model, feature.astype(np.float32), lambda1=0.01, lambda2=0.01)
        dag = self.get_adjacency(W_mlp)
        processed_cpdag = self.process_cpdag(dag)
        return dag, W_mlp, processed_cpdag
    
    def LF_pycausal(self, feature, algoId):
        '''
        Greedy Equivalence Search: Silander, T., & Myllymäki, P. (2006, July). A simple approach for finding the globally optimal Bayesian network structure. 
        In Proceedings of the Twenty-Second Conference on Uncertainty in Artificial Intelligence (pp. 445-452).
        https://github.com/bd2kccd/py-causal
        '''
        pc = pycausal()
        pc.start_vm()
        df = pd.DataFrame(feature)
        tetrad = s.tetradrunner()
        tetrad.run(algoId, dfs = df, scoreId = 'sem-bic', dataType = 'continuous',
                maxDegree = -1, faithfulnessAssumed = True, verbose = True)
        graph = tetrad.getTetradGraph()
        dot_str = pc.tetradGraphToDot(graph)
        matrix = np.zeros((feature.shape[1], feature.shape[1]))
        lines = dot_str.split("\n")
        for edge in lines[1:-1]:
            source=int(edge.split("->")[0].strip().rstrip()[1])
            dest = int(edge.split("->")[1].strip().rstrip()[1])
            relation = edge.split("->")[1].strip().rstrip()[5:-2].split(", ")
            for item in relation:
                if 'arrowhead' in item:
                    arrowhead = item.split('=')[1]
                elif 'arrowtail' in item:
                    arrowtail = item.split('=')[1]
            if arrowhead == 'none' and arrowtail == 'none':
                matrix[source,dest] = 0
            elif arrowhead == 'none' and arrowtail == 'normal':
                matrix[dest, source] = 1
            else:
                matrix[source,dest] = 1
        return matrix
    
    def LF_fci(self, feature, score_func=mv_fisherz, p_threshold=0.04):
        '''
        Fast Causal Inference: Spirtes, P., Meek, C., & Richardson, T. (1995, August). 
        Causal inference in the presence of latent variables and selection bias. In Proceedings of the Eleventh conference on Uncertainty in artificial intelligence (pp. 499-506)

        Score func can be either one of: fisherz, chisq, kci, gsq, mv_fisherz
        https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constrained-based%20causal%20discovery%20methods/FCI.html#id3 
        '''
        G = fci(feature, score_func, p_threshold, verbose=False)
        return self.get_adjacency(G[0].__dict__['graph'])
    
    def LF_pc(self, feature, score_func=mv_fisherz, p_threshold=0.04, uc_rule=0):
        '''
        PC algorithm: Spirtes, P., Glymour, C. N., Scheines, R., & Heckerman, D. (2000). Causation, prediction, and search. MIT press. 
        Score func can be either one of: fisherz, chisq, kci, gsq, mv_fisherz
        uc_rule can be 0,1,2 
        https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constrained-based%20causal%20discovery%20methods/PC.html
        '''
        cg = pc(feature, p_threshold, score_func, True, uc_rule, -1)
        return cg.G.dpath
    
    def LF_ICA_Lingam(self, feature):
        model = lingam.ICALiNGAM()
        model.fit(feature)
        return model.adjacency_matrix_

    def LF_Direct_Lingam(self, feature):
        model = lingam.DirectLiNGAM()
        model.fit(feature)
        return model.adjacency_matrix_

    def LF_Var_Lingam(self, feature):
        model = lingam.VARLiNGAM()
        model.fit(feature)
        return model.adjacency_matrices_[0]
    
    def LF_RCD(self, feature):
        model = lingam.RCD()
        model.fit(feature)
        return model.adjacency_matrix_
    
    def lf_SAM(self, feature):
        obj = SAM()
        output = obj.predict(pd.DataFrame(feature))
        return output

    def LF_bic_exact_search(self, feature):
        '''
        Exact Search: Silander, T., & Myllymäki, P. (2006, July). A simple approach for finding the globally optimal Bayesian network structure. 
        In Proceedings of the Twenty-Second Conference on Uncertainty in Artificial Intelligence (pp. 445-452).

        https://causal-learn.readthedocs.io/en/latest/search_methods_index/Score-based%20causal%20discovery%20methods/ExactSearch.html 
        '''
        dag_est, _ = bic_exact_search(feature)
        return dag_est
    
    def LF_lingam(self, feature):
        '''
        Lingam: https://sites.google.com/view/sshimizu06/lingam
        '''
        model = lingam.DirectLiNGAM()
        model.fit(feature)
        return model.adjacency_matrix_
    
    def LF_MMPC(self, feature):
        obj = MMPC()
        output = obj.predict(pd.DataFrame(feature))
        dag = retrieve_adjacency_matrix(output)
        return dag
    
    def LF_GS(self, feature):
        obj = GS()
        output = obj.predict(pd.DataFrame(feature))
        dag = retrieve_adjacency_matrix(output)
        return dag

    def LF_IAMB(self, feature):
        obj = IAMB()
        output = obj.predict(pd.DataFrame(feature))
        dag = retrieve_adjacency_matrix(output)
        return dag
    
    def LF_Inter_IAMB(self, feature):
        obj = Inter_IAMB()
        output = obj.predict(pd.DataFrame(feature))
        dag = retrieve_adjacency_matrix(output)
        return dag

# lf = LF()

