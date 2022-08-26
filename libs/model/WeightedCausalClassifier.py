import torch
import numpy as np
from tqdm import tqdm
from .CausalClassifier import CausalClassifier
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class WeightedCausalClassifier(CausalClassifier):
    def __init__(self):
        super(WeightedCausalClassifier, self).__init__()
        pass

    def get_points_weights_mask_once(self, train_data, model, feature_weights, batch_size=64, lr=1e-3, epochs=20, l2_penalty=0.1):
        trainloader, dataset, labels = self.features_to_dataloader(train_data, batch_size)
        model = model(dataset.shape[1], class_num=np.unique(labels).shape[0])
        model = self.train(model, trainloader, epochs=epochs, lr=lr, l2_weight=l2_penalty)
        _, _, f_x = self.evaluate(model, train_data, batch_size)
        
        points_weights = np.ones((train_data.shape[0], 1))
        causal_feature_idxs = np.argwhere(np.array(feature_weights) > 0.5).flatten()
        print(len(feature_weights))
        # print("causal_feature_idxs", causal_feature_idxs)
        print("train_data", train_data.shape)
        if len(causal_feature_idxs) > 0:
            print("CAUSAL FEATS LEN", len(causal_feature_idxs))
            masked_feats = np.copy(train_data)
            masked_feats[:,causal_feature_idxs] = 0
            # np.random.shuffle(masked_feats[:,causal_feature_idxs],)
            _, _, f_x_accent = self.evaluate(model, masked_feats, batch_size)
            f_x_diff_causal = np.sum(np.abs(f_x_accent - f_x), axis=1) #large for causal points

            points_weights = f_x_diff_causal 
        return points_weights.flatten()

    def get_points_weights(self, train_data, model, feature_weights, batch_size=64, lr=1e-3, epochs=20,):
        trainloader, dataset, labels = self.features_to_dataloader(train_data, batch_size)
        model = model(dataset.shape[1], class_num=np.unique(labels).shape[0])
        model = self.train(model, trainloader, epochs=epochs, lr=lr)
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

    def train(self, model, trainloader, epochs=10, lr = 1e-3, verbose=False, l2_weight = 0.1):
        optimizer = SGD(model.parameters(), lr, momentum=0.2)
        if cuda:
            model = model.cuda()
        model.train()
        for epoch in range(epochs):
            correct = 0
            for batch_idx, chunk in enumerate(trainloader):
                # print(chunk)
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
                correct += (predicted == target).sum()
                if verbose:
                    print('Epoch {}: train loss: {:.3f} accuracy{:.3f}'.format(epoch, loss.item(), float(correct*100) / float(self.batch_size*(batch_idx+1))))
                # Backward pass
                parameters = []
                for parameter in model.parameters():
                    parameters.append(parameter.view(-1))
                l2 = l2_weight * model.compute_l2_loss(torch.cat(parameters))
                loss += l2
                loss.backward()
                optimizer.step()
        return model
    
    def train_end_model(self, model, train_data, points_weights, batch_size=64, lr=1e-3, epochs=20, l2_weight=0.1, 
                        verbose=False):
        print("training end model...")
        trainloader, dataset, labels = self.features_to_dataloader(train_data, batch_size, points_weights)
        model = model(dataset.shape[1], class_num=np.unique(labels).shape[0])
        self.model = self.train(model, trainloader, epochs=epochs, lr=lr, verbose=verbose, l2_weight=l2_weight)
        return self.model