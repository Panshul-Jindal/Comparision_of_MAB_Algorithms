import random

class Agent():
    def __init__(self, n_arms):
        self.n_arms = n_arms

    def select_arm(self):
        raise NotImplementedError

    def update(self, chosen_arm, reward):
        raise NotImplementedError
    
class RandomAgent(Agent):
    def __init__(self, n_arms):
        super().__init__(n_arms)

    def select_arm(self):
        return random.randint(0, self.n_arms - 1)

    def update(self, chosen_arm, reward):
        pass