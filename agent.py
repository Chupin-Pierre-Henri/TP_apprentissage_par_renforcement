import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import random
from gym import wrappers, logger
from DQN import replay
from r_neurones import QNetwork

class Agent:
    def __init__(self, espace_action, espace_observation):
        self.espace_action = espace_action
        self.espace_observation = espace_observation
        self.qnetwork = QNetwork(espace_observation.shape[0], espace_action.n, 0, 32, 32)


    def act(self, ob, epsilon):
        ob = torch.from_numpy(ob).float().unsqueeze(0).to(self.device)
        q_actions = self.qnetwork(ob)
        if random.random() > epsilon:
            return np.argmax(q_actions.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_space.n))
