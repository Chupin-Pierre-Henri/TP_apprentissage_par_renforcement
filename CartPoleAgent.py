from Agent import Agent
from r_neuronne import rNeurones
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optimizer


class CartPoleAgent(Agent):
    
    def __init__(self, env, alpha=0.01, update_count=500, lr=0.005, from_file=False):

        super().__init__(env, alpha, update_count, lr)

        self.r_Neurones = rNeurones(self.espace_observation.shape[0], self.espace_action.n,64,64)
        self.reseau_cible = rNeurones(self.espace_observation.shape[0], self.espace_action.n,64,64)

        if from_file:
            self.r_Neurones.load_state_dict(torch.load("saved_params/cart_pole.pt"))

        self.loss_func = torch.nn.MSELoss()
        
        # Adam optimization is an extension to Stochastic gradient decent and can be used in
        # place of classical stochastic gradient descent to update network weights more efficiently.
        # pytorch documentation optimizer
        # https://pytorch.org/docs/stable/optim.html
        self.optimizer = torch.optim.SGD(self.r_Neurones.parameters(), lr=lr)
    
    def save_param(self):
        torch.save(self.r_Neurones.state_dict(), "saved_params/cart_pole.pt")