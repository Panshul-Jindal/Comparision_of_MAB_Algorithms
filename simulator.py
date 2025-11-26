import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# MAB simulator for an agent to interact with
class MAB():
    def __init__(self, n_arms, arm_distributions):
        self.n_arms = n_arms
        self.counts = [0] * n_arms  # Number of times each arm was played
        self.values = []  # values generated for each arm
        self.arm_distributions = arm_distributions  # True reward distributions for each arm    
    
    def play_arm(self, arm):
        """Simulate playing an arm and getting a reward."""
        rewards = []
        for iter_arm in range(self.n_arms):
            rewards.append(self.arm_distributions[iter_arm]())
        
        reward = rewards[arm]()

        # Updating agent's counts and Value history
        self.counts[arm] += 1
        self.values.append(rewards)

        return reward
    
class MAB_preload():
    def __init__(self, n_arms, arm_distributions, history_length=10):
        self.n_arms = n_arms
        self.counts = [0] * n_arms  # Number of times each arm
        self.values = []  # values generated for each arm
        self.arm_distributions = arm_distributions  # True reward distributions for each arm
        self.history_len = history_length

        self.upcoming_values = []
        self.preload_values(self.history_len)

        if len(arm_distributions) != n_arms:
            raise ValueError("Length of arm_distributions must be equal to n_arms")
        

    def preload_values(self, n_values):
        """Preload a set of values for each arm."""
        for iter_value in range(n_values):
            arm_values = []
            for iter_arm in range(self.n_arms):
                arm_values.append(self.arm_distributions[iter_arm]())
            
            self.upcoming_values.append(arm_values)

    def get_upcoming_rewards(self):
        """Get future rewards"""
        return self.upcoming_values[0]
    
    def play_arm(self, arm):
        """Simulate playing an arm and getting a reward."""
        if len(self.upcoming_values) == 0:
            self.preload_values(self.history_len)
        
        rewards = self.upcoming_values.pop(0)
        reward = rewards[arm]

        future_rewards = []
        for iter_arm in range(self.n_arms):
            future_rewards.append(self.arm_distributions[iter_arm]())
        self.upcoming_values.append(future_rewards)

        # Updating agent's counts and Value history
        self.counts[arm] += 1
        self.values.append(rewards)

        return reward
    
    def get_state(self, to_print=False):
        if to_print:
            print("Upcoming Values:")
            for idx, vals in enumerate(self.upcoming_values):
                print(f"Step {idx + 1}: {vals}")

        return np.array(self.upcoming_values)
    
    def display_state(self):
        fig, ax = plt.subplots()
        state = self.get_state()
        plt.imshow(state, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Reward Value')
        plt.xticks(range(self.n_arms))
        plt.xlabel('Arms')
        plt.ylabel('Future Rewards')
        plt.title('Upcoming Reward Values for Each Arm')
        plt.show()
    
    def display_state_with_choice(self, chosen_arm, time_offset=0, to_save=False, save_path=None, skip=False):
        triangle = Polygon([[chosen_arm - 0.1, -0.3], [chosen_arm + 0.1, -0.3], [chosen_arm, -0.1]], facecolor='black')

        fig, ax = plt.subplots()
        state = self.get_state()
        ax.add_patch(triangle)
        plt.imshow(state, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(label='Reward Value')
        plt.xticks(range(self.n_arms))
        plt.xlabel('Arms')
        plt.ylabel(f"Future Rewards ")
        plt.title(f"Upcoming Reward Values for Each Arm\n({time_offset} steps past)")
        if not to_save or save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()

if __name__ == "__main__":    
    import distributions

    # Example usage
    n_arms = 3
    arm_distributions = [
        distributions.Bernoulli(0.3),
        distributions.Bernoulli(0.8),
        distributions.Bernoulli(0.5)
    ]

    mab = MAB_preload(n_arms, arm_distributions, history_length=5)
    mab.display_state_with_choice(1,2)
    reward = mab.play_arm(1)
    print(f"Played arm 1, received reward: {reward}")
    mab.display_state()

    
    


