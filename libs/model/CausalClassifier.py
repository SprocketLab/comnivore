import numpy as np
import torch
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import copy
import torchvision

root_dir = "wilds_data"
cuda = True if torch.cuda.is_available() else False


FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class CausalClassifier:
    def __init__(self, G_causal=None):
        self.G_causal = G_causal
        if G_causal is not None:
            self.nodes_to_train = self.get_all_label_ancestors(G_causal, G_causal.shape[1]-1)
            self.unsure_nodes = np.copy(self.nodes_to_train)
            self.nodes_to_train = self.get_directly_dependent_nodes(G_causal)
            self.unsure_nodes = np.setdiff1d(self.unsure_nodes, self.nodes_to_train)

    def get_directly_dependent_nodes(self, G_causal):
        # does node share a parent with label? if no, put in direct_dependent_node
        direct_dependent_nodes = []
        for node in self.nodes_to_train:
            other_label_parents = np.delete(self.nodes_to_train, np.argwhere(self.nodes_to_train==node))
            node_parents = self.get_all_label_ancestors(G_causal, node)
            intersection = list(set(other_label_parents) & set(node_parents))
            if len(intersection) == 0:
                direct_dependent_nodes.append(node)
        return direct_dependent_nodes
    
    # def get_all_label_ancestors(self, G_causal, label_node):
    #     nodes = np.arange(G_causal.shape[1], dtype=int)[:-1]
    #     ancestors = []
    #     for node in tqdm(nodes):
    #         causal_path = nx.has_path(nx.DiGraph(G_causal),node,label_node)
    #         anti_causal_path = nx.has_path(nx.DiGraph(G_causal),label_node,node)
    #         if causal_path and not anti_causal_path:
    #             ancestors.append(node)
    #     return ancestors

    # @jit
    def get_all_label_ancestors(self, G_causal, label_node):
        assert G_causal.shape[0] == G_causal.shape[1]
        G_causal[-1,-1] = 0
        ancestors = np.argwhere(G_causal[:, label_node] != 0).flatten()
        edge_towards = np.argwhere(G_causal[label_node, :] != 0).flatten()
        edge_towards = np.array([e for e in edge_towards if e in ancestors])
        if len(edge_towards) > 0:
            ancestors = np.delete(ancestors, np.argwhere(ancestors==edge_towards))
        to_visit = np.copy(ancestors).tolist()
        visited = set([])
        while len(to_visit) > 0:
            # print(to_visit)
            curr_node = to_visit[0]
            parents = np.argwhere(G_causal[:, curr_node]).flatten()
            # is there any edge from the label node to ancestors? if yes, remove from list of ancestors 
            edge_towards = np.argwhere(G_causal[label_node, :] != 0).flatten()
            edge_towards = np.array([e for e in edge_towards if e in ancestors])
            if len(edge_towards) > 0:
                ancestors = np.delete(ancestors, np.argwhere(ancestors==edge_towards))
            parents = np.asarray([node for node in parents if node not in visited and node != label_node])
            ancestors = np.append(ancestors, parents)
            ancestors = np.unique(ancestors)
            if len(parents) > 0:
                to_visit = np.append(to_visit, parents)
            to_visit = np.unique(to_visit)
            visited.add(curr_node)
            to_visit = np.delete(to_visit, np.argwhere(to_visit == curr_node))
        return list(set(ancestors.astype(int).tolist()))

    def new_loss(self, y_pred, y_true):
        regular_loss_f = torch.nn.CrossEntropyLoss()
        regular_loss = regular_loss_f(y_pred, y_true)
        return regular_loss

    def train(self, model, trainloader, epochs=30, lr = 1e-3, verbose=False, l2_penalty=0.1, valdata=None, metadata_val=None, batch_size=32,\
        evaluate_func=None, log_freq=50, tune_by_metric='acc_wg'):
        # optimizer = SGD(model.parameters(), lr, momentum=0.9)
        optimizer = Adam(model.parameters(), lr, weight_decay=1.e-3)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader), eta_min=1.e-8)
        if cuda:
            model = model.cuda()
        model.train()
        val_perf = []
        best_chkpt = None
        best_val_perf = 0
        best_epoch = 0
        for epoch in tqdm(range(epochs)):
            correct = 0
            for batch_idx, (data, target) in enumerate(trainloader):
                if cuda:
                    data = data.cuda()
                    target = target.cuda()
                data = Variable(data.type(FloatTensor))
                target = Variable(target.type(LongTensor))
                optimizer.zero_grad()
                # Forward pass
                y_pred = model(data)
                y_pred = F.log_softmax(y_pred, dim=1)
            
                loss = self.new_loss(y_pred.squeeze(), target)
                # Compute Loss
                predicted = torch.max(y_pred.data, 1)[1] 
                correct += (predicted == target).sum()
                if verbose:
                    print('Epoch {}: train loss: {} accuracy{}'.format(epoch, loss.item(), float(correct*100) / float(self.batch_size*(batch_idx+1))))
                # Backward pass
                l2_weight = l2_penalty
                parameters = []
                for parameter in model.parameters():
                    parameters.append(parameter.view(-1))
                l2 = l2_weight * model.compute_l2_loss(torch.cat(parameters))
                loss += l2
                loss.backward()
                optimizer.step()
            if valdata is not None:
                outputs_val, labels_val, _ = self.evaluate(model, valdata, batch_size)
                results_obj_val, results_str_val = evaluate_func(outputs_val, labels_val, metadata_val)
                val_perf.append(results_obj_val)
                if results_obj_val[tune_by_metric] > best_val_perf:
                    best_val_perf = results_obj_val[tune_by_metric]
                    best_chkpt = copy.deepcopy(model)
                    best_epoch = epoch
                if (epoch+1) % log_freq == 0:
                    print(f"epoch: {epoch} Val \n {results_str_val}")
            scheduler.step()
        if best_chkpt is None:
            best_chkpt = copy.deepcopy(model)
        print("BEST EPOCH", best_epoch)
        return model, val_perf, best_chkpt

    def features_to_dataloader(self, data, batch_size, nodes_to_train=None, shuffle=True, generator=None):
        if nodes_to_train is None:
            self.nodes_to_train = [i for i in range(data.shape[1]-1)]
            nodes_to_train = self.nodes_to_train
        
        X = data[:, nodes_to_train]
        y = data[:, -1]
        tensor_x = torch.Tensor(X) # transform to torch tensor
        tensor_y = torch.Tensor(y)
        my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size, shuffle, generator=generator)
        return my_dataloader, X, y
        
    def train_baseline(self, model, train_data, batch_size=128, lr=1e-3, epochs=20, \
        verbose=False, n_layers=2, l2=0.1, dropout=0.1, valdata=None, metadata_val=None, generator=None, evaluate_func=None, log_freq=50,
        tune_by_metric='acc_wg'):
        
        self.nodes_to_train = [i for i in range(train_data.shape[1]-1)]
        trainloader, dataset, labels = self.features_to_dataloader(train_data, batch_size, self.nodes_to_train, generator=generator)
        self.batch_size = batch_size

        input_size = dataset.shape[1]
        output_size = np.unique(labels).shape[0]
        n_hidden = int((2/3) * (input_size + output_size))

        model = model(dataset.shape[1], class_num=np.unique(labels).shape[0], n_hidden=n_hidden, mlp_depth=n_layers, mlp_dropout=dropout)
        self.model, val_perf, self.best_chkpt = self.train(model, trainloader, epochs=epochs, lr=lr, \
            verbose=verbose, l2_penalty=l2, valdata=valdata, metadata_val=metadata_val, batch_size=batch_size, evaluate_func=evaluate_func, \
                log_freq=log_freq, tune_by_metric=tune_by_metric)
        return self.model, val_perf, self.best_chkpt

    def evaluate(self, model, test_data, batch_size=None, nodes_to_train=None):
        if batch_size == None:
            batch_size = self.batch_size
        if nodes_to_train is not None:
            testloader, _,_ = self.features_to_dataloader(test_data, batch_size, nodes_to_train, shuffle=False)
        else:
            testloader, _,_ = self.features_to_dataloader(test_data, batch_size, shuffle=False)
        correct = 0
        model.eval()
        y_preds = []
        labels = []
        outputs = []
        with torch.no_grad():
            for test_data, y_true in testloader:
                if cuda:
                    test_data = test_data.cuda()
                    y_true = y_true.cuda()
                output = model(test_data)
                
                y_pred = F.log_softmax(output, dim=1)
                
                outputs.append(y_pred.detach().cpu().numpy())
                y_pred = torch.argmax(y_pred, dim=1)
                y_preds.extend(y_pred.detach().cpu().numpy())
                labels.extend(y_true.detach().cpu().numpy())
                acc1 = np.argwhere((y_pred.detach().cpu().numpy() == y_true.detach().cpu().numpy())==True).shape[0]
                correct += acc1
        return np.asarray(y_preds), np.asarray(labels), np.vstack(outputs)