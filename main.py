import numpy as np
import random
import matplotlib.pyplot as plt
import os

import distributions
import simulator
import agents

import imageio as iio

BASE_RUN_DIR = r"Runs"

def makeLatestRunDirectory(base_dir=r"Runs/NamelessRuns"):
    """Get the latest run directory based on numeric names."""
    if not os.path.exists(base_dir):
        return None

    run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    if run_dirs == []:
        os.makedirs(os.path.join(base_dir, "1"))
        return os.path.join(base_dir, "1")

    latest_run = str(int(max(run_dirs, key=lambda x: int(x))) + 1)
    os.makedirs(os.path.join(base_dir, latest_run))
    return os.path.join(base_dir, latest_run)

def run(agent, mab, n_iterations, agent_name=None, mab_name=None):

    if agent_name is None or mab_name is None:
        if not os.path.exists(r"Runs/NamelessRuns"):
            os.makedirs(r"Runs/NamelessRuns")
            print(f"Created directory: NamelessRuns")
        latest_run_dir = makeLatestRunDirectory(r"Runs/NamelessRuns")
    else:
        if not os.path.exists(os.path.join(BASE_RUN_DIR, agent_name + "_" + mab_name)):
            os.makedirs(os.path.join(BASE_RUN_DIR, agent_name + "_" + mab_name))
            print(f"Created directory: {agent_name + '_' + mab_name}")
        latest_run_dir = makeLatestRunDirectory(os.path.join(BASE_RUN_DIR, agent_name + "_" + mab_name))

    rewards = []
    for iteration in range(n_iterations):
        chosen_arm = agent.select_arm()
        mab.display_state_with_choice(chosen_arm, time_offset=0, to_save=True,
                                     save_path=os.path.join(latest_run_dir, f"{iteration+1}.png"))
        reward = mab.play_arm(chosen_arm)
        agent.update(chosen_arm, reward)
        rewards.append(reward)

    image_files = sorted([os.path.join(latest_run_dir, f) for f in os.listdir(latest_run_dir) if f.endswith('.png')])

    images = [iio.imread(filename) for filename in image_files]

    iio.mimsave(os.path.join(latest_run_dir,'animation.gif'), images, fps=n_iterations//5) # fps controls the speed of the animation

    return rewards

if __name__ == "__main__":
    # Creating our MAB environment

    p1, p2, p3 = random.random(), random.random(), random.random()

    print("Bernoulli probabilities for arms:", p1, p2, p3)

    our_mab = simulator.MAB_preload(n_arms=3,
                            arm_distributions= 
                            (distributions.Bernoulli(p1),
                            distributions.Bernoulli(p2),
                            distributions.Bernoulli(p3)),
                            history_length=5
                            )

    # Creating a random agent

    our_agent = agents.UCB1Agent(n_arms=3)

    # Simulating interaction

    run(our_agent, our_mab, n_iterations=100, agent_name="UCB1", mab_name="BernoulliMAB")

    # TODO: optimise saving process, maybe optimise gif by directly creating gifs? or by storing data in a more efficient way

    """n_iterations = 1000
    for iteration in range(n_iterations):
        chosen_arm = our_agent.select_arm()
        reward = our_mab.play_arm(chosen_arm)
        our_agent.update(chosen_arm, reward)"""
