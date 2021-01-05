import torch
import torch.nn as nn
import torch.nn.functional as f


#ceci a été récupéré sur la documentation de pytorch il ne fonctionne pas car il faudrait changer des éléments
#nous n'avons pas eu le temps de le faire
class CnnModel(torch.nn.Module):    
    def __init__(self, h, w, output_size):
        super(CnnModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=2)
        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=7, stride=2)
        # self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=5, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 7, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 8

        self.head = torch.nn.Linear(linear_input_size, output_size)

        self.relu = torch.nn.functional.relu

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        return self.head(x.view(x.size(0), -1))