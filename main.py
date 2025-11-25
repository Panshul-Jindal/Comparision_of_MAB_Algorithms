import numpy as np
import random
import matplotlib.pyplot as plt

import distributions
import simulator
import agents

# Creating our MAB environment

p1, p2, p3 = random.random(), random.random(), random.random()

print("Bernoulli probabilities for arms:", p1, p2, p3)

our_mab = simulator.MAB(n_arms=3,
                        arm_distributions= 
                        (distributions.Bernoulli(p1),
                         distributions.Bernoulli(p2),
                         distributions.Bernoulli(p3))
                        )

# Creating a random agent

our_agent = agents.RandomAgent(n_arms=3)

# Simulating interaction

n_iterations = 1000
for iteration in range(n_iterations):
    chosen_arm = our_agent.select_arm()
    reward = our_mab.play_arm(chosen_arm)
    our_agent.update(chosen_arm, reward)
