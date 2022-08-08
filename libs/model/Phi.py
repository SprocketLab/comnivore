import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random

from torchvision import models
from torch.autograd import Variable

from transformers import CLIPProcessor, CLIPVisionModel

from sklearn.decomposition import PCA
from sklearn import datasets, cluster

from torch.utils.data import TensorDataset, DataLoader, Dataset

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

log_interval = 10
epochs = 20
workers=10

device = torch.device("cuda")


in_dim =28**2 *3
emb_dim=5
hidden_dim=1000
# z_hidden = 5

def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  

class Extractor_VAE:
    def __init__(self, z_hidden):
        self.model = self.VAE(z_hidden)
        self.z_hidden = z_hidden
    
    class VAE(nn.Module):
        def __init__(self, z_hidden):
            super(Extractor_VAE.VAE, self).__init__()

            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.fc21 = nn.Linear(hidden_dim, z_hidden)
            self.fc22 = nn.Linear(hidden_dim, z_hidden)
            self.fc3 = nn.Linear(z_hidden+emb_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, in_dim)

            self.embed = nn.Embedding(10, emb_dim)

        def encode(self, x):
            h1 = F.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)

        def decode(self, z, emb):
            lab_emb = self.embed(emb)
            h3 = F.relu(self.fc3(torch.cat((z,lab_emb), -1)))
            return torch.sigmoid(self.fc4(h3))

        def forward(self, x, emb):
            mu, logvar = self.encode(x.view(-1, in_dim))
            return self.decode(mu, emb), mu, logvar
        
        def extract_feature(self, x):
            feat, _ = self.encode(x.view(-1, in_dim))
            return feat

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, in_dim), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def train(self, train_loader, num_epochs=epochs, lr=1e-3):
        optimizer = optim.Adam(self.model.parameters(), lr)
        self.model.train()
        if cuda:
            self.model.to(device)
        train_loss = 0
        outputs = []
        for epoch in range(1, num_epochs + 1):
            for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
                if cuda:
                    data = data.to(device)
                    target = target.to(device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data, target)
                recon_image = recon_batch.view(data.shape[0], 3, 28, 28)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader.dataset)))
            outputs.append((epoch, data, recon_image))
        return outputs

class Extractor_CNN:
    def __init__(self, z_hidden, model='Vanilla_CNN', class_num=10):
        assert model in ['ResNet', 'Vanilla_CNN', 'Vgg']
        if model == 'ResNet':
            self.model = self.ResNet(z_hidden)
        elif model == 'Vanilla_CNN':
            self.model = self.Vanilla_CNN(z_hidden, class_num)
        self.z_hidden = z_hidden
    
    class ResNet(nn.Module):
        def __init__(self, z_hidden, class_num = 10):
            super(Extractor_CNN.ResNet, self).__init__()
            self.pretrained_model = models.resnet18(pretrained=True)
            num_ftrs = self.pretrained_model.fc.in_features
            self.feature_extractor = nn.Sequential(*list(self.pretrained_model.children())[:-1])
            self.fc1 = nn.Linear(num_ftrs, z_hidden)
            self.fc2 = nn.Linear(z_hidden, class_num)
        
        def forward(self, x):
            feature = self.feature_extractor(x)
            feature = torch.squeeze(feature)
            feature = F.relu(self.fc1(feature))
            out = self.fc2(feature)
            return out, feature
        
        def extract_feature(self, x):
            _, feat = self.forward(x)
            return feat
        
        def compute_l1_loss(self, w):
            return torch.abs(w).sum()
  
        def compute_l2_loss(self, w):
            return torch.square(w).sum()

    class Vanilla_CNN(nn.Module):
        def __init__(self, z_hidden, class_num = 10):
            super(Extractor_CNN.Vanilla_CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
            self.conv3 = nn.Conv2d(32,64, kernel_size=5)
            self.fc1 = nn.Linear(576, z_hidden)
            self.fc2 = nn.Linear(z_hidden, class_num)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(F.max_pool2d(self.conv3(x),2))
            x = F.dropout(x, p=0.5, training=self.training)
            x = torch.flatten(x, 1)
            feature = F.relu(self.fc1(x))
            # x = feature, training=self.training)
            x = self.fc2(feature)
            return x, feature
        
        def extract_feature(self, x):
            _, feat = self.forward(x)
            return feat
        
        def compute_l1_loss(self, w):
            return torch.abs(w).sum()
  
        def compute_l2_loss(self, w):
            return torch.square(w).sum()
        
    def train(self, train_loader, num_epochs=epochs, lr=1e-3):
        setup_seed(42)
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        error = nn.CrossEntropyLoss()
        self.model.train()
        if cuda:
            self.model.to(device)
        for epoch in tqdm(range(num_epochs)):
            correct = 0
            n = 0
            for batch_idx, (imgs, labels) in enumerate(train_loader):
                var_X_batch = Variable(imgs.type(FloatTensor))
                var_y_batch = Variable(labels.type(LongTensor))
                if cuda:
                    var_X_batch = var_X_batch.to(device)
                    var_y_batch = var_y_batch.to(device)
                batch_size = var_X_batch.shape[0]
                optimizer.zero_grad()
                output, _ = self.model(var_X_batch)
                output = F.log_softmax(output, dim=1)
                loss = error(output, var_y_batch)
                l1_weight = 0
                l2_weight = 0.15
                parameters = []
                for parameter in self.model.parameters():
                    parameters.append(parameter.view(-1))
                l1 = l1_weight * self.model.compute_l1_loss(torch.cat(parameters))
                l2 = l2_weight * self.model.compute_l2_loss(torch.cat(parameters))
                loss += l1
                loss += l2
                loss.backward()
                optimizer.step()

                # Total correct predictions
                predicted = torch.max(output.data, 1)[1] 
                correct += (predicted == var_y_batch).sum()
                n += var_y_batch.shape[0]

            print('====> Epoch: {} Average loss: {}, Acc:{}'.format(
                epoch, loss.item() / len(train_loader.dataset), (correct*100) / n))



class Extractor_CLIP:
    def __init__(self, z_hidden):
        self.z_hidden = z_hidden
        self.model = self.CLIP_Model()
        self.post_processor = self.FA_projector(z_hidden).project_
    
    class FA_projector:
        def __init__(self, z_hidden):
            self.fa = cluster.FeatureAgglomeration(n_clusters=z_hidden)
        
        def project_(self, X, y):
            projection = self.fa.fit_transform(X)
            mapping, components = self.get_fa_feature_mapping()
            return projection, mapping, components
        
        def get_fa_feature_mapping(self):
            cluster_labels = self.fa.labels_
            labels = np.unique(cluster_labels)
            mapping = {}
            for label_ in labels:
                top_incluencing_feat = np.argwhere(cluster_labels == label_).flatten()
                mapping[label_] = top_incluencing_feat
            return mapping, cluster_labels
    
    class PCA_projector:
        def __init__(self, z_hidden):
            self.pca = PCA(z_hidden)
        
        def project_(self, X, y =None):
            projection = self.pca.fit_transform(X)
            mapping, components = self.get_pc_feature_mapping()
            return projection, mapping, components
        
        def get_pc_feature_mapping(self, epsilon=0.1):
            components = np.abs(self.pca.components_)
            n_components, n_features = components.shape
            mapping = {}
            for pc_idx in range(n_components):
                pc_component = components[pc_idx]
                top_incluencing_feat = np.argwhere(pc_component > epsilon).flatten()
                mapping[pc_idx] = top_incluencing_feat
            return mapping, components

    class CLIP_Model:
        def __init__(self):
            # self.projector = projector
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def eval(self):
            pass
        
        def get_image_as_list(self, dataset):
            images = []
            for (imgs) in dataset:
                images.append((imgs.detach().cpu().numpy()))
            return images
        
        def extract_feature(self, images):
            img_list = self.get_image_as_list(images)
            outputs = self.extract_feature_batch(img_list)
            return torch.Tensor(np.asarray(outputs)).cuda()

        def extract_feature_batch(self, x):
            with torch.no_grad():
                inputs = self.processor(images=x, return_tensors="pt")
                outputs = self.model(inputs.pixel_values)
                return outputs.pooler_output.detach().cpu().numpy()

class Phi:
    def __init__(self, extractor):
        #extractor should be a trained Extractor object
        self.extractor = extractor

    def get_z_features(self, dataloader, domainbed=False):
        self.extractor.model.eval()
        with torch.no_grad():
            for i, chunk in tqdm(enumerate(dataloader)):
                data = chunk[0]
                target = chunk[1]
                data = data.to(device)
                target = target.to(device)
                z = self.extractor.model.extract_feature(data)
                data_matrix = torch.hstack((z, target.view(-1,1)))
                if i == 0:
                    features = np.zeros((1, data_matrix.shape[1]))
                features = np.vstack((features, data_matrix.detach().cpu().numpy()))
            features = np.delete(features, 0, 0)
            if self.extractor.post_processor:
                X_processed, feature_mapping, components = self.extractor.post_processor(features[:, :-1], features[:, -1])
                features_projected = np.hstack((X_processed, features[:, -1].reshape(-1,1)))
            return features, features_projected, feature_mapping, components
    