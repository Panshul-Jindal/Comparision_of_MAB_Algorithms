import random
import math


class Agent():
    def __init__(self, n_arms):
        self.n_arms = n_arms

    def select_arm(self):
        raise NotImplementedError

    def update(self, chosen_arm, reward):
        raise NotImplementedError
    def give_instance_values(self):
        return {}
    
class RandomAgent(Agent):
    def __init__(self, n_arms):
        super().__init__(n_arms)

    def select_arm(self):
        return random.randint(0, self.n_arms - 1)

    def update(self, chosen_arm, reward):
        pass

class EpsilonGreedyAgent(Agent):
    def __init__(self, n_arms, epsilon=0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon
        self.counts = [0] * n_arms
        self.est_means = [0.0] * n_arms

    def select_arm(self):
        if random.random() > self.epsilon:
            return self.est_means.index(max(self.est_means))
        else:
            return random.randint(0, self.n_arms - 1)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.est_means[chosen_arm]
        # Update the estimated value using incremental formula
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.est_means[chosen_arm] = new_value

    def give_instance_values(self):
        return {
            "est_means": self.est_means,
            "counts": self.counts,
            "epsilon": self.epsilon
        }


class EpsilonDecreasingAgent(Agent):
    def __init__(self, n_arms, initial_epsilon=1.0, min_epsilon=0.1, decay_rate=0.99):
        super().__init__(n_arms)
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.counts = [0] * n_arms
        self.est_means = [0.0] * n_arms

    def select_arm(self):
        if random.random() > self.epsilon:
            return self.est_means.index(max(self.est_means))
        else:
            return random.randint(0, self.n_arms - 1)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.est_means[chosen_arm]
        # Update the estimated value using incremental formula
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.est_means[chosen_arm] = new_value
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def give_instance_values(self):
        return {
            "est_means": self.est_means,
            "counts": self.counts,
            "epsilon": self.epsilon
        }

class ExplorationFirstAgent(Agent):
    def __init__(self, n_arms, exploration_rounds=100):
        super().__init__(n_arms)
        self.exploration_rounds = exploration_rounds
        self.total_counts = 0
        self.counts = [0] * n_arms
        self.est_means = [0.0] * n_arms

    def select_arm(self):
        if self.total_counts < self.exploration_rounds:
            return random.randint(0, self.n_arms - 1)
        else:
            return self.est_means.index(max(self.values))

    def update(self, chosen_arm, reward):
        self.total_counts += 1
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.est_means[chosen_arm]
        # Update the estimated value using incremental formula
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.est_means[chosen_arm] = new_value

    def give_instance_values(self):
        return {
            "est_means": self.est_means,
            "counts": self.counts,
            "exploration_rounds": self.exploration_rounds,
            "total_counts": self.total_counts
        }

class UCB1Agent(Agent):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.counts = [0] * n_arms
        self.confidence_intervals = [0.0] * n_arms
        self.ucb_values = [0.0] * n_arms
        self.est_means = [0.0] * n_arms
        self.total_counts = 0

    def select_arm(self):
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        for arm in range(self.n_arms):
            confidence_interval = (2 * (math.log(self.total_counts) / self.counts[arm])) ** 0.5
            self.ucb_values[arm] = self.est_means[arm] + confidence_interval
            self.confidence_intervals[arm] = confidence_interval
        return self.ucb_values.index(max(self.ucb_values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        ucb_value = self.ucb_values[chosen_arm]
        est_mean = self.est_means[chosen_arm]
        confidence_interval = self.confidence_intervals[chosen_arm]
        # Update the estimated value using incremental formula
        self.est_means[chosen_arm] = ((n - 1) / n) * est_mean + (1 / n) * reward
        self.confidence_intervals[chosen_arm] = (2 * (math.log(n-1) / n)) ** 0.5
        self.ucb_values[chosen_arm] = self.est_means[chosen_arm] + self.confidence_intervals[chosen_arm]
    
    def give_instance_values(self):
        return {
            "est_means": self.est_means,
            "counts": self.counts,
            "ucb_values": self.ucb_values,
            "confidence_intervals": self.confidence_intervals,
            "total_counts": self.total_counts
        }

class UCB2Agent(Agent):
    def __init__(self, n_arms, alpha=0.5):
        super().__init__(n_arms)
        self.alpha = alpha
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_counts = 0
        self.current_rounds = [0] * n_arms

    def select_arm(self):
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            r = self.current_rounds[arm]
            bonus = ((1 + self.alpha) * math.log(math.e * self.total_counts / (r + 1))) / (2 * r)
            bonus = bonus ** 0.5
            ucb_values[arm] = self.values[arm] + bonus
        return ucb_values.index(max(ucb_values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        self.current_rounds[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Update the estimated value using incremental formula
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

class UCBTunedAgent(Agent):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_counts = 0

    def select_arm(self):
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            bonus = (2 * (math.log(self.total_counts) / self.counts[arm])) ** 0.5
            ucb_values[arm] = self.values[arm] + bonus
        return ucb_values.index(max(ucb_values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Update the estimated value using incremental formula
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

class MossAgent(Agent):
    def __init__(self, n_arms, total_rounds=1000):
        super().__init__(n_arms)
        self.total_rounds = total_rounds
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_counts = 0

    def select_arm(self):
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        moss_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            bonus = math.sqrt((max(math.log(self.total_rounds / (self.n_arms * self.counts[arm])), 0)) / self.counts[arm])
            moss_values[arm] = self.values[arm] + bonus
        return moss_values.index(max(moss_values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Update the estimated value using incremental formula
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

class KLUCBAgent(Agent):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_counts = 0

    def kl_divergence(self, p, q):
        if p == 0:
            return 0
        if p == 1:
            return float('inf') if q < 1 else 0
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    def select_arm(self):
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        kl_ucb_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            upper_bound = 1.0
            lower_bound = self.values[arm]
            while upper_bound - lower_bound > 1e-6:
                mid = (upper_bound + lower_bound) / 2
                if self.kl_divergence(self.values[arm], mid) <= (math.log(self.total_counts) / self.counts[arm]):
                    lower_bound = mid
                else:
                    upper_bound = mid
            kl_ucb_values[arm] = lower_bound
        return kl_ucb_values.index(max(kl_ucb_values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Update the estimated value using incremental formula
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

class BayesUCBAgent(Agent):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.successes = [0] * n_arms
        self.failures = [0] * n_arms
        self.total_counts = 0

    def select_arm(self):
        ucb_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            alpha = self.successes[arm] + 1
            beta = self.failures[arm] + 1
            ucb_values[arm] = random.betavariate(alpha, beta)
        return ucb_values.index(max(ucb_values))

    def update(self, chosen_arm, reward):
        self.total_counts += 1
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1

class ThompsonSamplingAgent(Agent):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.successes = [0] * n_arms
        self.failures = [0] * n_arms

    def select_arm(self):
        sampled_theta = [random.betavariate(self.successes[arm] + 1, self.failures[arm] + 1) for arm in range(self.n_arms)]
        return sampled_theta.index(max(sampled_theta))

    def update(self, chosen_arm, reward):
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1

    