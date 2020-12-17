import torch
import torch.nn as nn
import torch.nn.functional as f


class QNetwork(nn.Module):
    
    """
    dimension_etat (int): le dimension des etat exemple la taille est de 4 pour CartPole-v1
    dimension_action (int): le dimension des actions exemple la taille est de 2 pour CartPole-v1
    seed (int): un seed random
    fc_unit (int): Number of nodes in first hidden layer
    """
    def __init__(self, dimension_etat, dimension_action, seed, fc_unit=32):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc = nn.Linear(dimension_etat, fc_unit)
        self.fc_end = nn.Linear(fc_unit,dimension_action)


    def forward(self, x):
        x = f.relu(self.fc(x))
        return self.fc_end(x)
