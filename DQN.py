from collections import namedtuple
import random


class replay(object):

    def __init__(self, cap):
        self.cap = cap
        self.memoire = []
        self.position = 0

    def push(self, transition):
        if len(self.memoire) < self.cap:
            self.memoire.append(transition)
        else:
            self.memoire[self.position] = transition
        self.position = (self.position + 1) % self.cap

    
    def sample(self, batch_size):
        return random.sample(self._memory, batch_size))


    def __len__(self):
        return len(self.memoire)
