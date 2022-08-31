import torch
import numpy as np
from tqdm import tqdm
from .CausalClassifier import CausalClassifier
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import copy

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class WeightedCausalClassifier(CausalClassifier):
    def __init__(self):
        super(WeightedCausalClassifier, self).__init__()
        pass

    def get_base_predictor(self, train_data, model, feature_weights, \
                                    batch_size=64, lr=1e-3, epochs=20, l2_penalty=0.1, \
                                    evaluate_func=None, metadata_val=None, \
                                    valdata=None, log_freq=20):
        print("training base predictor...")
        self.batch_size = batch_size
        trainloader, dataset, labels = self.features_to_dataloader(train_data, batch_size)
        model = model(dataset.shape[1], class_num=np.unique(labels).shape[0])
        
        _, self.best_chkpt = self.train(model, trainloader, epochs=epochs, lr=lr, l2_weight=l2_penalty, verbose=True, 
                           evaluate_func=evaluate_func, metadata_val=metadata_val, valdata=valdata,\
                               batch_size=batch_size, log_freq=log_freq)
        
        _, _, f_x = self.evaluate(self.best_chkpt, train_data, batch_size)
        return self.best_chkpt, f_x
    
    def get_points_weights_with_base_predictor(self, base_predictor, f_x, train_data, feature_weights, batch_size):
        print("getting points weights...")
        points_weights = np.ones((train_data.shape[0], 1))
        causal_weight_thr = 0.5
        causal_feature_idxs = np.argwhere(np.array(feature_weights) > causal_weight_thr).flatten()
        if len(causal_feature_idxs) > 0:
            masked_feats = np.copy(train_data)
            masked_feats[:,causal_feature_idxs] = 0
            
            _, _, f_x_accent = self.evaluate(base_predictor, masked_feats, batch_size)
            f_x_diff_causal = np.sum(np.abs(f_x_accent - f_x), axis=1) #large for causal points

            points_weights = f_x_diff_causal 
            return points_weights.flatten()
        return None

    def get_points_weights_mask_once(self, train_data, model, feature_weights, \
                                    batch_size=64, lr=1e-3, epochs=20, l2_penalty=0.1, \
                                    evaluate_func=None, metadata_val=None, \
                                    valdata=None, log_freq=20):
        print("getting points weights...")
        self.batch_size = batch_size
        base_predictor, f_x = self.get_base_predictor(train_data, model, feature_weights, \
                                    batch_size, lr, epochs, l2_penalty, \
                                    evaluate_func, metadata_val, \
                                    valdata, log_freq)
        
        points_weights = np.ones((train_data.shape[0], 1))
        causal_weight_thr = 0.5
        causal_feature_idxs = np.argwhere(np.array(feature_weights) > causal_weight_thr).flatten()
        if len(causal_feature_idxs) > 0:
            masked_feats = np.copy(train_data)
            masked_feats[:,causal_feature_idxs] = 0
            
            _, _, f_x_accent = self.evaluate(self.best_chkpt, masked_feats, batch_size)
            f_x_diff_causal = np.sum(np.abs(f_x_accent - f_x), axis=1) #large for causal points

            points_weights = f_x_diff_causal 
            return points_weights.flatten()
        return None

    def get_points_weights(self, train_data, model, feature_weights, batch_size=64, lr=1e-3, epochs=20, l2_penalty=0.1, \
        evaluate_func=None, metadata_val=None, valdata=None, log_freq=20):
        trainloader, dataset, labels = self.features_to_dataloader(train_data, batch_size)
        model = model(dataset.shape[1], class_num=np.unique(labels).shape[0])
        model = self.train(model, trainloader, epochs=epochs, lr=lr, l2_weight=l2_penalty, verbose=True, 
                           evaluate_func=evaluate_func, metadata_val=metadata_val, valdata=valdata,\
                               batch_size=batch_size, log_freq=log_freq)
        _, _, f_x = self.evaluate(model, train_data, batch_size)
        points_weights = np.zeros((train_data.shape[0],2))
        print("getting point weights...")
        for f_idx in tqdm(range(train_data.shape[1]- 1)):
            f_weight = feature_weights[f_idx]
            masked_feats = np.copy(train_data)
            masked_feats[:,f_idx] = 0
            _, _, f_x_accent = self.evaluate(model, masked_feats, batch_size)
            f_x_diff = np.abs(f_x_accent - f_x)
            if f_weight < 0.5: # if feature is noncausal => weight down the point => get higher point score
                points_weights += (1-f_weight) * f_x_diff
            # else:
            #     points_weights += 0 * f_x_diff
        points_weights = np.sum(points_weights, axis=1)
        points_weights[points_weights==0] = 1
        return points_weights

    def new_loss(self, y_pred, y_true, batch_weights=None):
        if batch_weights is not None:
            regular_loss_f = torch.nn.CrossEntropyLoss(reduction='none')
            regular_loss = regular_loss_f(y_pred, y_true) #--> shape batch_size x 1
            regular_loss = regular_loss * batch_weights
            return regular_loss.mean()
            # regular_loss = torch.mean(regular_loss)
        else:
            regular_loss_f = torch.nn.CrossEntropyLoss()
            regular_loss = regular_loss_f(y_pred, y_true)
            return regular_loss

    def train(self, model, trainloader, epochs=10, lr = 1e-3, verbose=False, l2_weight = 0.1, evaluate_func=None, \
        metadata_val=None, valdata=None, batch_size=64, log_freq=20, tune_by_metric='acc_wg'):
        optimizer = SGD(model.parameters(), lr, momentum=0.8)
        # optimizer = Adam(model.parameters(), lr, weight_decay=1.e-5)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader), eta_min=1.e-8)
        if cuda:
            model = model.cuda()
        model.train()
        val_perf = []
        best_val_perf = 0
        best_chkpt = None
        best_epoch = 0
        for epoch in tqdm(range(epochs)):
            for _, chunk in enumerate(trainloader):
                if len(chunk) == 3:
                    data, target, weights = chunk
                else:
                    data, target = chunk
                    weights = None
                if cuda:
                    data = data.cuda()
                    target = target.cuda()
                    if weights is not None:
                        weights = weights.cuda()
                data = Variable(data.type(FloatTensor))
                target = Variable(target.type(LongTensor))
                optimizer.zero_grad()
                # Forward pass
                y_pred = model(data)
                y_pred = F.log_softmax(y_pred, dim=1)
            
                loss = self.new_loss(y_pred.squeeze(), target, weights)
                # Compute Loss
                predicted = torch.max(y_pred.data, 1)[1] 
                # Backward pass
                parameters = []
                for parameter in model.parameters():
                    parameters.append(parameter.view(-1))
                l2 = l2_weight * model.compute_l2_loss(torch.cat(parameters))
                loss += l2
                loss.backward()
                optimizer.step()
            if valdata is not None:
                outputs_, labels_, _ = self.evaluate(model, valdata, batch_size)
                results_obj_, results_str_ = evaluate_func(outputs_, labels_, metadata_val)
                val_perf.append(results_obj_)
                if results_obj_[tune_by_metric] > best_val_perf:
                    best_val_perf = results_obj_[tune_by_metric]
                    best_chkpt = copy.deepcopy(model)
                    best_epoch = epoch
                if verbose and (epoch+1)%log_freq == 0:
                    print(f"Epoch: {epoch} \n {results_str_}")
            # scheduler.step()        
        if best_chkpt is None:
            best_chkpt = copy.deepcopy(model)
        if valdata is not None:
            print("BEST EPOCH", best_epoch)
        return model, best_chkpt
    
    def train_end_model(self, model, train_data, evaluate_func, points_weights=None, valdata=None, metadata_val=None,\
                        batch_size=64, lr=1e-3, epochs=20, l2_weight=0.1, verbose=False, log_freq=20, tune_by_metric='acc_wg'):
        print("training end model...")
        self.batch_size = batch_size
        if points_weights is not None:
            trainloader, dataset, labels = self.features_to_dataloader(train_data, batch_size, points_weights)
        else:
            trainloader, dataset, labels = self.features_to_dataloader(train_data, batch_size)
        model = model(dataset.shape[1], class_num=np.unique(labels).shape[0])
        self.model, self.best_chkpt = self.train(model, trainloader, epochs=epochs, lr=lr, l2_weight=l2_weight, verbose=verbose, \
                                                evaluate_func=evaluate_func, metadata_val=metadata_val, valdata=valdata,\
                                                batch_size=batch_size, log_freq=log_freq, tune_by_metric=tune_by_metric)
        return self.model, self.best_chkpt
    
    def features_to_dataloader(self, data, batch_size=64, points_weights=[], shuffle=True):
        points_weights = np.array(points_weights)
        X = data[:, :-1]
        y = data[:, -1]
        tensor_x = torch.Tensor(X) # transform to torch tensor
        tensor_y = torch.Tensor(y)
        if len(points_weights) == 0:
            my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        else:
            my_dataset = TensorDataset(tensor_x,tensor_y, torch.Tensor(points_weights).reshape(-1,1))
        my_dataloader = DataLoader(my_dataset, batch_size, shuffle) 
        return my_dataloader, X, y