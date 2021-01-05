import torch
import torch.nn as nn
import torch.nn.functional as f

class rNeurones(torch.nn.Module):
    
    #def __init__(self, input_size, output_size, hidden_layer_size=None):
    def __init__(self, dimension_etat, dimension_action, fc_1=64, fc_2=32):
        super(rNeurones, self).__init__()
        self.fc1 = torch.nn.Linear(dimension_etat, fc_1)
        self.fc2 = torch.nn.Linear(fc_1, fc_2)
        self.fc_end = torch.nn.Linear(fc_2, dimension_action)


    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return self.fc_end(x)