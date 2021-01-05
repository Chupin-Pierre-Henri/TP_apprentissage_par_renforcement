from collections import namedtuple
import random


Transition = namedtuple(typename='Transition', field_names=['state', 'action', 'next_state', 'reward', 'done'])

class replay(object):

    def __init__(self, cap):
        self.cap = cap
        self.memoire = []
        self.position = 0

    def push(self, value):
        if len(self.memoire) < self.cap:
            self.memoire.append(value)
        else:
            self.memoire[self.position] = value
        self.position = (self.position + 1) % self.cap

    
    def sample(self, batch_size):
        return random.sample(self.memoire, batch_size)



    def __len__(self):
        return len(self.memoire)
