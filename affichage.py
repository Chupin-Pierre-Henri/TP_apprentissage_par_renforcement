import matplotlib.pyplot as plt


class Affichage():
    
    def __init__(self) -> None:
        self.rewards = []

    def start_episode(self):
        self.c = 0
        self.r_sum = 0

    def add_value(self, reward):
        self.c += 1
        self.r_sum += reward

    def recorde_episode(self):
        self.rewards.append(self.r_sum)

    def show(self):
        plt.plot(self.rewards)
        plt.ylabel('rewards par episode')
        plt.xlabel('episodes')
        plt.show()