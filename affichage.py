import matplotlib.pyplot as plt


class Affichage():
    
    def __init__(self) -> None:
        self.recompense = []

    def start_episode(self):
        self.c = 0
        self.r_sum = 0

    def add_value(self, recompense):
        self.c += 1
        self.r_sum += recompense

    def recorde_episode(self):
        self.recompense.append(self.r_sum)

    def show(self):
        plt.plot(self.recompense)
        plt.ylabel('recompense par episode')
        plt.xlabel('episodes')
        plt.show()