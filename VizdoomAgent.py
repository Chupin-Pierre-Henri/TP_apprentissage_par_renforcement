import gym
from Agent import Agent
from ConvuNeuronne import CnnModel
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optimizer
import skimage
import numpy as np



class VizDoomAgent(Agent):
    
    def __init__(self, env: gym.Env, alpha=0.01, u_c=500, eps=1.0, lr=0.005, res=(42, 24, 3), from_file=False): # (112, 64, 3) (28, 16, 3)
        super().__init__(env, alpha, u_c, lr)
        self.resolution = res

        self.r_Neurones = CnnModel(res[0], res[1], self.espace_action.n)
        self.reseau_cible = CnnModel(res[0], res[1], self.espace_action.n)

        if from_file:
            self.r_Neurones.load_state_dict(torch.load("saved_params/vizdoom.pt"))

        self.reseau_cible.load_state_dict(self.r_Neurones.state_dict())

        self.loss_func = torch.nn.MSELoss() # the mean squared error
        self.optimizer = torch.optim.SGD(self.r_Neurones.parameters(), lr=lr)


    def act(self, ob, reward, done):
        # print(ob.shape)
        return super().act(self.preprocess(ob), reward, done)

    #donné dans le sujet
    def preprocess(self, img):
        img = img[0]
        img = skimage.transform.resize(img, self.resolution)
        #passage en noir et blanc
        img = skimage.color.rgb2gray(img)
        #passage en format utilisable par pytorch
        img = img.astype(np.float32)
        # print(img.shape)
        img = img.reshape((1, self.resolution[0], self.resolution[1]))
        return img

    def save_param(self):
        torch.save(self.r_Neurones.state_dict(), "saved_params/vizdoom.pt")