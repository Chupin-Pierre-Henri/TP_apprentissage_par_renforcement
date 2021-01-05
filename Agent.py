import gym
import DQN
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class Agent():
    
    def __init__(self, env, alpha, update_count, lr,from_file=False):
        self.espace_observation = env.observation_space # espace des états
        self.espace_action = env.action_space # espace des actions
        if from_file:
            self.EPS_START = 0.004
        else:            
            self.EPS_START = 1
        self.EPS_DECAY = 0.997
        self.EPS_END = 0.004
        self.alpha = alpha
        self.update_count = update_count
        self.steps_done = 0

    def greedy_exploration(self, qvalues):

        if np.random.rand() < self.EPS_START:   
            return self.espace_action.sample()

        action = torch.argmax(qvalues)
        return int(action)

    def preprocess(self, ob):
        return ob

    def save(self, path):
        torch.save(self.r_Neurones.state_dict(), path)

    def act(self, ob, reward, done):
        if self.EPS_START > self.EPS_END:
            self.EPS_START *= self.EPS_DECAY
        inputs = torch.tensor([ob]).float()

        self.r_Neurones.eval()
        with torch.no_grad():  
            qvaleurs = self.r_Neurones(inputs)
        self.r_Neurones.train()

        return self.greedy_exploration(qvaleurs)

    def train(self, experiences, horizon,viz = False):
        if not viz:
            s = []
            a = []
            s_next = []
            a_next = []
            d = []
            for e in experiences:
                if e is not None:
                    s.append(e.state)
                    a.append(e.action)
                    s_next.append(e.next_state)
                    a_next.append(e.reward)
                    d.append(e.done)
            s_vstack = np.vstack(s)
            a_vstack = np.vstack(a)
            s_nextvstack = np.vstack(s_next)
            a_nextvstack = np.vstack(a_next)
            d_vstack = np.vstack(d)

            etats = torch.from_numpy(s_vstack).float()
            actions = torch.from_numpy(a_vstack).long()
            etats_suivant = torch.from_numpy(s_nextvstack).float()
            recompense = torch.from_numpy(a_nextvstack).float()
            dones = torch.from_numpy(d_vstack).float()
        #j'avais un problème de dimension pour vizdoom on m'a montré cet solution qui règle mon problème je n'ai pas eu le temps 
        # de bien comprendre comment cela fonctionné (j'écris cela le 05/01 à 21h47)
        else:
            etats = torch.tensor([ s.state for s in experiences ]).float()
            etats_suivant = torch.tensor([ s.next_state for s in experiences ]).float()
            dones = torch.tensor([ 0 if s.done else 1 for s in experiences ], dtype=torch.int8).unsqueeze(1)
            recompense = torch.tensor([ s.reward for s in experiences ]).unsqueeze(1)
            actions = torch.tensor([ s.action for s in experiences ]).unsqueeze(1)
        
        self.r_Neurones.train()
        self.reseau_cible.eval()
        self.optimizer.zero_grad()

        

        with torch.no_grad():
            labels_next = self.reseau_cible(etats_suivant).detach().max(1)[0].unsqueeze(1)

        # exemple de la doc pytorch pour l'utilisation du MSELoss
        # >>> loss = nn.MSELoss()
        # >>> input = torch.randn(3, 5, requires_grad=True)
        # >>> target = torch.randn(3, 5)
        # >>> output = loss(input, target)
        # >>> output.backward()
        loss = torch.nn.MSELoss()
        prediction = self.r_Neurones(etats).gather(1, actions)
        labels = recompense + (horizon * labels_next * (1 - dones))

        output = loss(prediction, labels)
        output.backward()
        self.optimizer.step()

        for t_param,l_param in zip(self.reseau_cible.parameters(), self.r_Neurones.parameters()):
            t_param.data.copy_(self.alpha * l_param.data + (1 - self.alpha) * t_param.data)