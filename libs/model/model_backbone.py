import torch 
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, input_size, class_num, n_hidden):
        super(MLP, self).__init__()
        self.class_num = class_num
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(self.input_size, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, class_num)
        self.relu = torch.nn.ReLU()
        # self.fc3 = torch.nn.Linear(int(n_hidden/2), class_num)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc3(x)
        return x
    
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()
  
    def compute_l2_loss(self, w):
        return torch.square(w).sum()

class CLIPMLP(torch.nn.Module):
    """Just  an MLP"""
    def __init__(self, input_size, class_num, n_hidden=1024, mlp_depth=2, mlp_dropout=0.1, add_residual=False,
                 add_norm=False):
        super(CLIPMLP, self).__init__()
        self.input = torch.nn.Linear(input_size, n_hidden)
        self.add_norm = add_norm
        if self.add_norm:
            self.bn_in = torch.nn.BatchNorm1d(n_hidden)
        self.dropout = torch.nn.Dropout(mlp_dropout)
        self.hiddens = torch.nn.ModuleList([
            torch.nn.Linear(n_hidden, n_hidden)
            for _ in range(mlp_depth-2)])
        if self.add_norm:
            self.bn_hids = torch.nn.ModuleList([
                torch.nn.BatchNorm1d(n_hidden) for _ in range(mlp_depth-2)])
        self.output = torch.nn.Linear(n_hidden, class_num)
        self.add_residual = add_residual
        if self.add_norm:
            self.bn_out = torch.nn.BatchNorm1d(class_num)
        self.n_outputs = class_num
        self.n_inputs = input_size

    def forward(self, x):
        orig_x = x
        x = self.input(x)
        if self.add_norm:
            x = self.bn_in(x)
        x = self.dropout(x)
        x = F.relu(x)
        for idx, hidden in enumerate(self.hiddens):
            x = hidden(x)
            if self.add_norm:
                x = self.bn_hids[idx](x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        if self.add_norm:
            x = self.bn_out(x)
        if self.add_residual:
            if self.n_outputs == self.n_inputs:
                x = x + orig_x
            else:
                assert self.n_outputs == (2 * self.n_inputs)
                x[..., :self.n_inputs] += orig_x
        return x
    
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()
  
    def compute_l2_loss(self, w):
        return torch.square(w).sum()