import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time

import distributions
import simulator
import agents

import imageio.v2 as iio

import copy

BASE_RUN_DIR = r"Runs"
BASE_PLOT_DIR = r"Plots"

def exp_indices(n, f):
    return np.unique(np.round(np.logspace(0, np.log10(n), f))).astype(int)

def makeLatestRunDirectory(base_dir=r"Runs/NamelessRuns"):
    """Get the latest run directory based on numeric names."""
    if not os.path.exists(base_dir):
        os.makedirs(os.path.join(base_dir, "1"))
        return os.path.join(base_dir, "1")

    run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    if run_dirs == []:
        os.makedirs(os.path.join(base_dir, "1"))
        return os.path.join(base_dir, "1")

    latest_run = str(int(max(run_dirs, key=lambda x: int(x))) + 1)
    os.makedirs(os.path.join(base_dir, latest_run))
    return os.path.join(base_dir, latest_run)

def run(agent, mab, n_iterations, agent_name=None, mab_name=None, instance_id=None, skipAnim=False):

    if not skipAnim:
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
            if instance_id is not None:
                latest_run_dir = os.path.join(latest_run_dir, f"{instance_id}")
                os.makedirs(latest_run_dir)

    instance_internal_info = []
    choices = []
    arm1_reward_seq = []
    arm1_reward_seq_sum = []
    arm2_reward_seq = []
    arm2_reward_seq_sum = []
    arm3_reward_seq = []
    arm3_reward_seq_sum = []
    agent_reward_seq = []
    agent_reward_seq_sum = []
    agent_rew_sum = 0
    arm1_rew_sum = 0
    arm2_rew_sum = 0
    arm3_rew_sum = 0
    regret_seq = []
    FRAMES_TO_SAVE = 20
    SECONDS = 5
    skip_step = n_iterations // FRAMES_TO_SAVE if n_iterations >= FRAMES_TO_SAVE else 1
    #iterations_to_show = set(range(0, n_iterations, skip_step))
    iterations_to_show = set(exp_indices(n_iterations, FRAMES_TO_SAVE-1))
    iterations_to_show.add(n_iterations-1)


    for iteration in range(n_iterations):
        chosen_arm = agent.select_arm()
        choices.append(chosen_arm)
        mab_reward_tuple = mab.get_upcoming_rewards()
        if iteration in iterations_to_show and not skipAnim:
            mab.display_state_with_choice(chosen_arm, time_offset=iteration, to_save=True,
                                        save_path=os.path.join(latest_run_dir, f"{iteration+1}.png"))
        reward = mab.play_arm(chosen_arm)
        agent.update(chosen_arm, reward)

        # Instance info
        instance_iter_info = agent.get_instance_values().items()
        instance_iter_info = {k: np.array(v) for k, v in instance_iter_info}
        instance_internal_info.append(instance_iter_info)

        agent_reward_seq.append(reward)
        arm1_reward_seq.append(mab_reward_tuple[0])
        arm2_reward_seq.append(mab_reward_tuple[1])
        arm3_reward_seq.append(mab_reward_tuple[2])

        arm1_rew_sum += mab_reward_tuple[0]
        arm2_rew_sum += mab_reward_tuple[1]
        arm3_rew_sum += mab_reward_tuple[2]
        agent_rew_sum += reward

        arm1_reward_seq_sum.append(arm1_rew_sum)
        arm2_reward_seq_sum.append(arm2_rew_sum)
        arm3_reward_seq_sum.append(arm3_rew_sum)
        agent_reward_seq_sum.append(agent_rew_sum)


        best_sum = max(arm1_rew_sum, arm2_rew_sum, arm3_rew_sum)

        regret = best_sum - agent_rew_sum
        regret_seq.append(regret)

     
    

    if not skipAnim:
        image_files = sorted([os.path.join(latest_run_dir, f) for f in os.listdir(latest_run_dir) if f.endswith('.png')])
        image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        images = [iio.imread(filename) for filename in image_files]

        iio.mimsave(os.path.join(latest_run_dir,'animation.gif'), images, fps=FRAMES_TO_SAVE/SECONDS) # fps controls the speed of the animation


        fig, axes = plt.subplots(2,2)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        axes = axes.flatten()
        axes[0].plot(range(1,n_iterations+1), regret_seq)
        axes[0].set_xlabel("Iterations")
        axes[0].set_ylabel("Regret")
        axes[0].set_title(f"Regret vs Iterations")
        
        axes[1].plot(range(1,n_iterations+1), arm1_reward_seq_sum, color='r', label="Arm 1")
        axes[1].plot(range(1,n_iterations+1), arm2_reward_seq_sum, color='g', label="Arm 2")
        axes[1].plot(range(1,n_iterations+1), arm3_reward_seq_sum, color='b', label="Arm 3")
        axes[1].set_xlabel("Iterations")
        axes[1].set_ylabel("Hidden Arm Rewards")
        axes[1].legend(fontsize=4)
        axes[1].set_title(f"Reward vs Iterations Arm")

        
        axes[2].plot(range(1,n_iterations+1), agent_reward_seq_sum, color='black', label="Agent")
        axes[2].set_xlabel("Iterations")
        axes[2].set_ylabel("Agent Rewards")
        axes[2].set_title(f"Reward vs Iterations Agent")

        if best_sum == arm1_rew_sum:
            best_arm = 0
        elif best_sum == arm2_rew_sum:
            best_arm = 1
        else:
            best_arm = 2

        correct = [int(best_arm == choice) for choice in choices]
        percent = []
        sum = 0
        for idx in range(n_iterations):
            num = correct[idx]
            sum += num
            percent.append(sum/(idx+1))
        
        axes[3].plot(range(1,n_iterations+1), percent)
        axes[3].set_xlabel("Iterations")
        axes[3].set_ylabel("Percentage")
        axes[3].set_title("Percentage of Best Arm Chosen by Agent")

        for ax in axes:
            ax.title.set_fontsize(12)
            ax.xaxis.label.set_size(10)
            ax.yaxis.label.set_size(10)
            ax.tick_params(axis='both', labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        
        fig.suptitle(f"Results for {agent_name}_{mab_name}_{instance_id}", fontsize=14)
        fig.savefig(os.path.join(latest_run_dir,'result.jpg'), dpi=900)
        plt.close(fig)
        


    return np.array(agent_reward_seq), instance_internal_info, np.array(agent_reward_seq_sum), np.array(regret_seq)

if __name__ == "__main__":
    INSTANCES_TO_RUN = 100
    ITERATION_COUNT = 2500
    # Creating our MAB environment


    mab_close = simulator.MAB_preload(n_arms=3,
                            arm_distributions= 
                            (distributions.Bernoulli(0.4),
                            distributions.Bernoulli(0.37),
                            distributions.Bernoulli(0.43)),
                            history_length=5
                            )

    mab_far = simulator.MAB_preload(n_arms=3,
                            arm_distributions=
                            (distributions.Bernoulli(0.4),
                            distributions.Bernoulli(0.2),
                            distributions.Bernoulli(0.6)),
                            history_length=5
                            )

        
    # Creating agents

    randomAgentCreator       = lambda x : agents.RandomAgent(n_arms=3)
    epsilonGreedyAgentCreator       = lambda x : agents.EpsilonGreedyAgent(n_arms=3)
    epsilonDecreasingAgentCreator   = lambda x : agents.EpsilonDecreasingAgent(n_arms=3)
    ExplorationFirstAgentCreator    = lambda x : agents.ExplorationFirstAgent(n_arms=3, m=ITERATION_COUNT//10)
    ucb1AgentCreator                = lambda x : agents.UCB1Agent(n_arms=3)
    ucb2AgentCreator                = lambda x : agents.UCB2Agent(n_arms=3)
    ucbTunedAgentCreator            = lambda x : agents.UCBTunedAgent(n_arms=3)
    mossAgentCreator                = lambda x : agents.MossAgent(n_arms=3, horizon=ITERATION_COUNT)
    kl_ucbAgentCreator              = lambda x : agents.KLUCBAgent(n_arms=3)
    bayesianUCBAgentCreator         = lambda x : agents.BayesUCBAgent(n_arms=3)
    thompsonAgentCreator            = lambda x : agents.ThompsonSamplingAgent(n_arms=3)


    agents_to_test = [
        ("Epsilon-Greedy", epsilonGreedyAgentCreator),
        ("Epsilon-Decreasing", epsilonDecreasingAgentCreator),
        ("Exploration-First", ExplorationFirstAgentCreator),
        ("UCB1", ucb1AgentCreator),
        ("UCB2", ucb2AgentCreator),
        ("UCB-Tuned", ucbTunedAgentCreator),
        ("MOSS", mossAgentCreator),
        ("KL-UCB", kl_ucbAgentCreator),
        ("Bayesian UCB", bayesianUCBAgentCreator),
        ("Thompson Sampling", thompsonAgentCreator)]

    # Simulating interaction for FAR mab
    sim_start_time = time.time()

    # FAR

    print("Starting Simulation on Far")
    for agent_name, agent_maker in agents_to_test[:]:
        print(f"Running simulations for Agent: {agent_name} on Far MAB")
        agent_start_time = time.time()
        agent_rewards_all_instances = []
        agent_info_all_instances = []
        agent_rew_sum_all = []
        agent_regret_seq_all = []
        skip_stuff = INSTANCES_TO_RUN//5
        if skip_stuff < 1:
            skip_stuff = 1
        for instance_id in range(INSTANCES_TO_RUN):
            curr_agent = agent_maker(None)
            instance_rewards, instance_info, instance_rew_sum, instance_regret_seq = run(curr_agent, 
                                                                                            mab_far, 
                                                                                            n_iterations=ITERATION_COUNT, 
                                                                                            agent_name=f"{agent_name}", 
                                                                                            mab_name="BernoulliMABFar", 
                                                                                            instance_id=instance_id,
                                                                                            skipAnim=(instance_id%skip_stuff != 0))
            
            agent_rewards_all_instances.append(instance_rewards)
            agent_info_all_instances.append(instance_info)
            agent_rew_sum_all.append(instance_rew_sum)
            agent_regret_seq_all.append(instance_regret_seq)
        
        keys = list(agent_info_all_instances[0][0].keys())

        agent_info = []
        for iteration in range(ITERATION_COUNT):
            iter_info = {}
            for key in keys:
                iter_info[key] = np.mean([agent_info_all_instances[inst_id][iteration][key] for inst_id in range(INSTANCES_TO_RUN)], axis=0)
            agent_info.append(iter_info)

        fig, axes = plt.subplots(2, int(np.ceil(len(keys)/2)), figsize=(5 * int(np.ceil(len(keys) / 2)), 10))
        axes = axes.flatten()

        colors = ['r', 'g', 'b']  # or use any colormap
        labels = ['Arm 1', 'Arm 2', 'Arm 3']  # adjust based on meaning

        for key_id in range(len(keys)):
            key = keys[key_id]
            values = [agent_info[iteration][key] for iteration in range(ITERATION_COUNT)]
            ax = axes[key_id]

            # Plot each series with color and label
            for i in range(len(values[0])):  # assuming 3 values in each iteration
                ax.plot(
                    [values[j][i] for j in range(ITERATION_COUNT)], 
                    label=labels[i], 
                    color=colors[i]
                )

            # Add title and legend
            ax.set_title(f"{key} for {agent_name} Agent")
            ax.set_xlabel("Iterations")
            ax.set_ylabel(key)
            ax.legend()

        agent_plot_dir = os.path.join(BASE_PLOT_DIR, agent_name+"_"+"Far")
        plot_dir = makeLatestRunDirectory(agent_plot_dir)
        info_plot_path = os.path.join(plot_dir, "Info")



        plt.savefig(info_plot_path,dpi=900)
        plt.close()
          
        mean_agent_regret = np.mean(np.array(agent_regret_seq_all), axis = 0)
        
        plt.figure()
        plt.plot(range(1, ITERATION_COUNT+1), mean_agent_regret)
        plt.xlabel("Iterations")
        plt.ylabel("Regret")
        plt.title(f"Average Regret of {agent_name} on BernoulliMAB Far")
        plt.savefig(os.path.join(plot_dir, "Regret"), dpi=900)
        plt.close()    


    # CLOSE
    for agent_name, agent_maker in agents_to_test[:]:
        print(f"Running simulations for Agent: {agent_name} on Near MAB")
        agent_start_time = time.time()
        agent_rewards_all_instances = []
        agent_info_all_instances = []
        agent_rew_sum_all = []
        agent_regret_seq_all = []
        skip_stuff = INSTANCES_TO_RUN//5
        if skip_stuff < 1:
            skip_stuff = 1
        for instance_id in range(INSTANCES_TO_RUN):
            curr_agent = agent_maker(None)
            instance_rewards, instance_info, instance_rew_sum, instance_regret_seq = run(curr_agent, 
                                                                                            mab_close, 
                                                                                            n_iterations=ITERATION_COUNT, 
                                                                                            agent_name=f"{agent_name}", 
                                                                                            mab_name="BernoulliMABNear", 
                                                                                            instance_id=instance_id,
                                                                                            skipAnim=(instance_id%skip_stuff != 0))
            
            agent_rewards_all_instances.append(instance_rewards)
            agent_info_all_instances.append(instance_info)
            agent_rew_sum_all.append(instance_rew_sum)
            agent_regret_seq_all.append(instance_regret_seq)
        
        keys = list(agent_info_all_instances[0][0].keys())

        agent_info = []
        for iteration in range(ITERATION_COUNT):
            iter_info = {}
            for key in keys:
                iter_info[key] = np.mean([agent_info_all_instances[inst_id][iteration][key] for inst_id in range(INSTANCES_TO_RUN)], axis=0)
            agent_info.append(iter_info)

        fig, axes = plt.subplots(2, int(np.ceil(len(keys)/2)), figsize=(5 * int(np.ceil(len(keys) / 2)), 10))
        axes = axes.flatten()

        colors = ['r', 'g', 'b']  # or use any colormap
        labels = ['Arm 1', 'Arm 2', 'Arm 3']  # adjust based on meaning

        for key_id in range(len(keys)):
            key = keys[key_id]
            values = [agent_info[iteration][key] for iteration in range(ITERATION_COUNT)]
            ax = axes[key_id]

            # Plot each series with color and label
            for i in range(len(values[0])):  # assuming 3 values in each iteration
                ax.plot(
                    [values[j][i] for j in range(ITERATION_COUNT)], 
                    label=labels[i], 
                    color=colors[i]
                )

            # Add title and legend
            ax.set_title(f"{key} for {agent_name} Agent")
            ax.set_xlabel("Iterations")
            ax.set_ylabel(key)
            ax.legend()

        agent_plot_dir = os.path.join(BASE_PLOT_DIR, agent_name+"_"+"Near")
        plot_dir = makeLatestRunDirectory(agent_plot_dir)
        info_plot_path = os.path.join(plot_dir, "Info")



        plt.savefig(info_plot_path,dpi=900)
        plt.close()
          
        mean_agent_regret = np.mean(np.array(agent_regret_seq_all), axis = 0)
        
        plt.figure()
        plt.plot(range(1, ITERATION_COUNT+1), mean_agent_regret)
        plt.xlabel("Iterations")
        plt.ylabel("Regret")
        plt.title(f"Average Regret of {agent_name} on BernoulliMAB Near")
        plt.savefig(os.path.join(plot_dir, "Regret"), dpi=900)
        plt.close()    

        
    print(f"Simulation completed in {time.time() - sim_start_time:.2f} seconds.")

    

    """n_iterations = 1000
    for iteration in range(n_iterations):
        chosen_arm = our_agent.select_arm()
        reward = our_mab.play_arm(chosen_arm)
        our_agent.update(chosen_arm, reward)"""
