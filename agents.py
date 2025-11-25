import random
import math
from scipy import stats


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
    """Random baseline agent - selects arms uniformly at random"""
    def __init__(self, n_arms):
        super().__init__(n_arms)

    def select_arm(self):
        return random.randint(0, self.n_arms - 1)

    def update(self, chosen_arm, reward):
        pass


class EpsilonGreedyAgent(Agent):
    """
    Epsilon-Greedy Algorithm (Fixed ε)
    Reference: Algorithm box 1 in document
    """
    def __init__(self, n_arms, epsilon=0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon
        # Initialization: N_a(0) = 0, μ̂_a(0) = 0 for all arms
        self.counts = [0] * n_arms  # N_a(t)
        self.est_means = [0.0] * n_arms  # μ̂_a(t)

    def select_arm(self):
        """
        Algorithm Step 1: With probability (1-ε), select argmax μ̂_a(t-1)
                         With probability ε, select uniformly at random
        """
        if random.random() > self.epsilon:
            # Exploit: choose best arm (greedy)
            return self.est_means.index(max(self.est_means))
        else:
            # Explore: choose random arm
            return random.randint(0, self.n_arms - 1)

    def update(self, chosen_arm, reward):
        """
        Algorithm Step 3: Update counts and estimated means
        N_A_t(t) = N_A_t(t-1) + 1
        μ̂_A_t(t) = μ̂_A_t(t-1) + (1/N_A_t(t))(X_t - μ̂_A_t(t-1))
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        old_mean = self.est_means[chosen_arm]
        # Incremental mean update formula
        self.est_means[chosen_arm] = old_mean + (1.0 / n) * (reward - old_mean)

    def give_instance_values(self):
        return {
            "est_means": self.est_means,
            "counts": self.counts,
            "epsilon": self.epsilon
        }


class EpsilonDecreasingAgent(Agent):
    """
    Epsilon-Decreasing Algorithm
    Reference: Algorithm box 2 in document
    """
    def __init__(self, n_arms, c=1.0):
        super().__init__(n_arms)
        self.c = c  # Constant for epsilon schedule
        # Initialization: N_a(0) = 0, μ̂_a(0) = 0
        self.counts = [0] * n_arms
        self.est_means = [0.0] * n_arms
        self.t = 0  # Current round

    def select_arm(self):
        """
        Algorithm Step 1: Compute ε_t = min{1, cK/t}
        Algorithm Step 2: With probability (1-ε_t), exploit; with probability ε_t, explore
        """
        self.t += 1
        # Compute exploration probability: ε_t = min{1, cK/t}
        epsilon_t = min(1.0, (self.c * self.n_arms) / self.t)
        
        if random.random() > epsilon_t:
            # Exploit: select best arm
            return self.est_means.index(max(self.est_means))
        else:
            # Explore: select uniformly at random
            return random.randint(0, self.n_arms - 1)

    def update(self, chosen_arm, reward):
        """
        Algorithm Step 3: Update N_a(t) and μ̂_a(t) as in fixed-ε algorithm
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        old_mean = self.est_means[chosen_arm]
        self.est_means[chosen_arm] = old_mean + (1.0 / n) * (reward - old_mean)

    def give_instance_values(self):
        epsilon_t = min(1.0, (self.c * self.n_arms) / max(1, self.t))
        return {
            "est_means": self.est_means,
            "counts": self.counts,
            "epsilon_t": epsilon_t,
            "t": self.t
        }


class ExplorationFirstAgent(Agent):
    """
    Explore-Then-Commit (Exploration-First) Algorithm
    Reference: Algorithm box 3 in document
    """
    def __init__(self, n_arms, m=10):
        super().__init__(n_arms)
        self.m = m  # Plays per arm in exploration phase
        # Initialization: N_a(0) = 0, μ̂_a(0) = 0
        self.counts = [0] * n_arms
        self.est_means = [0.0] * n_arms
        self.total_counts = 0
        self.exploration_done = False
        self.best_arm = None

    def select_arm(self):
        """
        Exploration phase: Play each arm exactly m times
        Commit phase: After mK rounds, play a* = argmax μ̂_a(mK) forever
        """
        # Exploration phase: ensure each arm is played m times
        if not self.exploration_done:
            for arm in range(self.n_arms):
                if self.counts[arm] < self.m:
                    return arm
            # Exploration complete, commit to best arm
            self.exploration_done = True
            self.best_arm = self.est_means.index(max(self.est_means))
        
        # Commit phase: always play best arm
        return self.best_arm

    def update(self, chosen_arm, reward):
        """
        Update: N_a(t) = N_a(t-1) + 1
                μ̂_a(t) = μ̂_a(t-1) + (1/N_a(t))(X_t - μ̂_a(t-1))
        """
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        old_mean = self.est_means[chosen_arm]
        self.est_means[chosen_arm] = old_mean + (1.0 / n) * (reward - old_mean)

    def give_instance_values(self):
        return {
            "est_means": self.est_means,
            "counts": self.counts,
            "m": self.m,
            "exploration_done": self.exploration_done,
            "best_arm": self.best_arm,
            "total_counts": self.total_counts
        }


class UCB1Agent(Agent):
    """
    UCB1 Algorithm
    Reference: Algorithm box 4 in document
    """
    def __init__(self, n_arms):
        super().__init__(n_arms)
        # Initialization: Play each arm once (done in select_arm)
        self.counts = [0] * n_arms  # N_a(t)
        self.est_means = [0.0] * n_arms  # μ̂_a(t)
        self.total_counts = 0

    def select_arm(self):
        """
        Initialization: Play each arm once
        Algorithm Step 1: Compute U_a(t) = μ̂_a(t-1) + sqrt(2*log(t) / N_a(t-1))
        Algorithm Step 2: Select A_t ∈ argmax_a U_a(t)
        """
        # Initialization: play each arm once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # Compute UCB index for each arm
        ucb_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            # U_a(t) = μ̂_a(t-1) + sqrt(2*log(t) / N_a(t-1))
            exploration_bonus = math.sqrt((2 * math.log(self.total_counts)) / self.counts[arm])
            ucb_values[arm] = self.est_means[arm] + exploration_bonus
        
        # Select arm with maximum UCB value
        return ucb_values.index(max(ucb_values))

    def update(self, chosen_arm, reward):
        """
        Algorithm Step 3: Update N_A_t(t) and μ̂_A_t(t)
        N_A_t(t) = N_A_t(t-1) + 1
        μ̂_A_t(t) = μ̂_A_t(t-1) + (1/N_A_t(t))(X_t - μ̂_A_t(t-1))
        """
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        old_mean = self.est_means[chosen_arm]
        self.est_means[chosen_arm] = old_mean + (1.0 / n) * (reward - old_mean)
    
    def give_instance_values(self):
        ucb_values = []
        for arm in range(self.n_arms):
            if self.counts[arm] > 0:
                bonus = math.sqrt((2 * math.log(max(1, self.total_counts))) / self.counts[arm])
                ucb_values.append(self.est_means[arm] + bonus)
            else:
                ucb_values.append(float('inf'))
        
        return {
            "est_means": self.est_means,
            "counts": self.counts,
            "ucb_values": ucb_values,
            "confidence_intervals": [ucb - est for ucb, est in zip(ucb_values, self.est_means)],
            "total_counts": self.total_counts
        }


class UCB2Agent(Agent):
    """
    UCB2 Algorithm
    Reference: Algorithm box 5 in document
    """
    def __init__(self, n_arms, alpha=0.5):
        super().__init__(n_arms)
        self.alpha = alpha  # Parameter α > 0
        # Initialization: N_a(0) = 0, r_a = 0, μ̂_a(0) = 0
        self.counts = [0] * n_arms  # N_a(t)
        self.est_means = [0.0] * n_arms  # μ̂_a(t)
        self.r = [0] * n_arms  # Epoch counter r_a for each arm
        self.total_counts = 0
        self.current_arm = None  # Arm currently in phase
        self.phase_pulls_remaining = 0  # Pulls remaining in current phase

    def _tau(self, r):
        """Definition: τ(r) = ⌈(1+α)^r⌉"""
        return math.ceil((1 + self.alpha) ** r)

    def select_arm(self):
        """
        Algorithm Step 1: For every arm with N_a(t-1) = 0, play it once to initialize
        Algorithm Step 2: Compute U_a(t) = μ̂_a(t-1) + sqrt((1+α)*log(e*t/τ(r_a)) / (2*τ(r_a)))
        Algorithm Step 3: Let a* ∈ argmax_a U_a(t)
        Algorithm Step 4: Play arm a* repeatedly until N_a*(t) = τ(r_a* + 1)
        """
        # If currently in a phase, continue pulling the current arm
        if self.phase_pulls_remaining > 0:
            self.phase_pulls_remaining -= 1
            return self.current_arm
        
        # Step 1: Initialize any arm that hasn't been played
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                self.current_arm = arm
                self.phase_pulls_remaining = 0
                return arm
        
        # Step 2: Compute UCB2 index for each arm
        ucb2_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            tau_r = self._tau(self.r[arm])
            # U_a(t) = μ̂_a(t-1) + sqrt((1+α)*log(e*t/τ(r_a)) / (2*τ(r_a)))
            log_term = math.log(math.e * self.total_counts / tau_r)
            exploration_bonus = math.sqrt(((1 + self.alpha) * log_term) / (2 * tau_r))
            ucb2_values[arm] = self.est_means[arm] + exploration_bonus
        
        # Step 3: Select arm with maximum UCB2 value
        self.current_arm = ucb2_values.index(max(ucb2_values))
        
        # Step 4: Set up phase - play this arm until N_a = τ(r_a + 1)
        tau_next = self._tau(self.r[self.current_arm] + 1)
        self.phase_pulls_remaining = tau_next - self.counts[self.current_arm] - 1  # -1 for current pull
        
        return self.current_arm

    def update(self, chosen_arm, reward):
        """
        Update μ̂_a incrementally after each reward
        At end of phase: set r_a ← r_a + 1
        """
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        old_mean = self.est_means[chosen_arm]
        self.est_means[chosen_arm] = old_mean + (1.0 / n) * (reward - old_mean)
        
        # If phase just completed, increment epoch counter
        if self.phase_pulls_remaining == 0 and self.counts[chosen_arm] == self._tau(self.r[chosen_arm] + 1):
            self.r[chosen_arm] += 1

    def give_instance_values(self):
        return {
            "est_means": self.est_means,
            "counts": self.counts,
            "ucb2_values": [
                self.est_means[arm] + math.sqrt(((1 + self.alpha) * math.log(math.e * self.total_counts / self._tau(self.r[arm]))) / (2 * self._tau(self.r[arm])))
                if self.counts[arm] > 0 else float('inf')
                for arm in range(self.n_arms)
            ],
            "bonuses": [
                math.sqrt(((1 + self.alpha) * math.log(math.e * self.total_counts / self._tau(self.r[arm]))) / (2 * self._tau(self.r[arm])))
                if self.counts[arm] > 0 else float('inf')
                for arm in range(self.n_arms)
            ],
            "r": self.r,
            "alpha": self.alpha,
            "current_arm": self.current_arm,
            "phase_pulls_remaining": self.phase_pulls_remaining
        }


class UCBTunedAgent(Agent):
    """
    UCB1-Tuned Algorithm
    Reference: Algorithm box 6 in document
    """
    def __init__(self, n_arms):
        super().__init__(n_arms)
        # Initialization: Play each arm once to get μ̂_a(1) and m̂_a^(2)(1)
        self.counts = [0] * n_arms  # N_a(t)
        self.est_means = [0.0] * n_arms  # μ̂_a(t)
        self.sum_squares = [0.0] * n_arms  # For computing empirical second moment
        self.total_counts = 0

    def select_arm(self):
        """
        Initialization: Play each arm once
        Algorithm Step 1: Compute σ̂_a^2(t-1) = m̂_a^(2)(t-1) - (μ̂_a(t-1))^2
                         V_a(t-1) = σ̂_a^2(t-1) + sqrt(2*log(t) / N_a(t-1))
        Algorithm Step 2: Compute U_a(t) = μ̂_a(t-1) + sqrt((log(t)/N_a(t-1)) * min{1/4, V_a(t-1)})
        Algorithm Step 3: Select A_t ∈ argmax_a U_a(t)
        """
        # Initialization: play each arm once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # Compute UCB-Tuned index for each arm
        ucb_tuned_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            n = self.counts[arm]
            mean = self.est_means[arm]
            
            # Step 1: Compute empirical variance estimate
            # σ̂_a^2 = m̂_a^(2) - (μ̂_a)^2, where m̂_a^(2) is the empirical second moment
            second_moment = self.sum_squares[arm] / n
            variance = second_moment - (mean ** 2)
            variance = max(0, variance)  # Ensure non-negative
            
            # V_a(t-1) = σ̂_a^2 + sqrt(2*log(t) / N_a)
            V = variance + math.sqrt((2 * math.log(self.total_counts)) / n)
            
            # Step 2: Compute UCB-Tuned index
            # U_a(t) = μ̂_a + sqrt((log(t)/N_a) * min{1/4, V_a})
            exploration_term = math.sqrt((math.log(self.total_counts) / n) * min(0.25, V))
            ucb_tuned_values[arm] = mean + exploration_term
        
        # Step 3: Select arm with maximum UCB-Tuned value
        return ucb_tuned_values.index(max(ucb_tuned_values))

    def update(self, chosen_arm, reward):
        """
        Algorithm Step 3: Update N_a, μ̂_a, and m̂_a^(2) incrementally
        """
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        old_mean = self.est_means[chosen_arm]
        
        # Update mean incrementally
        self.est_means[chosen_arm] = old_mean + (1.0 / n) * (reward - old_mean)
        
        # Update sum of squares for variance computation
        self.sum_squares[chosen_arm] += reward ** 2

    def give_instance_values(self):
        return {
            "est_means": self.est_means,
            "counts": self.counts,
            "ucb_tuned_values": [
                self.est_means[arm] + math.sqrt((math.log(self.total_counts) / self.counts[arm]) * min(0.25, 
                    max(0, (self.sum_squares[arm] / self.counts[arm]) - (self.est_means[arm] ** 2) + 
                    math.sqrt((2 * math.log(self.total_counts)) / self.counts[arm]))))
                if self.counts[arm] > 0 else float('inf')
                for arm in range(self.n_arms)
            ],
            "bonuses": [
                math.sqrt((math.log(self.total_counts) / self.counts[arm]) * min(0.25, 
                    max(0, (self.sum_squares[arm] / self.counts[arm]) - (self.est_means[arm] ** 2) + 
                    math.sqrt((2 * math.log(self.total_counts)) / self.counts[arm]))))
                if self.counts[arm] > 0 else float('inf')
                for arm in range(self.n_arms)
            ],
            "sum_squares": self.sum_squares,
            "total_counts": self.total_counts
        }


class MossAgent(Agent):
    """
    MOSS Algorithm (Minimax Optimal Strategy in Stochastic)
    Reference: Algorithm box 7 in document
    """
    def __init__(self, n_arms, horizon):
        super().__init__(n_arms)
        self.horizon = horizon  # T (total number of rounds)
        # Initialization: Play each arm once
        self.counts = [0] * n_arms  # N_a(t)
        self.est_means = [0.0] * n_arms  # μ̂_a(t)
        self.total_counts = 0

    def select_arm(self):
        """
        Initialization: Play each arm once
        Algorithm Step 1: For each arm, compute
                         U_a(t) = μ̂_a(t-1) + sqrt(max{0, log(T/(K*N_a(t-1)))} / N_a(t-1))
        Algorithm Step 2: Select A_t ∈ argmax_a U_a(t)
        """
        # Initialization: play each arm once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # Step 1: Compute MOSS index for each arm
        moss_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            n = self.counts[arm]
            # U_a(t) = μ̂_a + sqrt(max{0, log(T/(K*N_a))} / N_a)
            log_term = max(0, math.log(self.horizon / (self.n_arms * n)))
            exploration_bonus = math.sqrt(log_term / n)
            moss_values[arm] = self.est_means[arm] + exploration_bonus
        
        # Step 2: Select arm with maximum MOSS value
        return moss_values.index(max(moss_values))

    def update(self, chosen_arm, reward):
        """
        Algorithm Step 3: Update N_a and μ̂_a as usual
        """
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        old_mean = self.est_means[chosen_arm]
        self.est_means[chosen_arm] = old_mean + (1.0 / n) * (reward - old_mean)

    def give_instance_values(self):
        return {
            "est_means": self.est_means,
            "counts": self.counts,
            "moss_values": [
                self.est_means[arm] + math.sqrt(max(0, math.log(self.horizon / (self.n_arms * self.counts[arm]))) / self.counts[arm])
                if self.counts[arm] > 0 else float('inf')
                for arm in range(self.n_arms)
            ],
            "bonuses": [
                math.sqrt(max(0, math.log(self.horizon / (self.n_arms * self.counts[arm]))) / self.counts[arm])
                if self.counts[arm] > 0 else float('inf')
                for arm in range(self.n_arms)
            ],
            "horizon": self.horizon,
            "total_counts": self.total_counts
        }


class KLUCBAgent(Agent):
    """
    KL-UCB Algorithm (Bounded Rewards)
    Reference: Algorithm box 8 in document
    """
    def __init__(self, n_arms, c=3.0):
        super().__init__(n_arms)
        self.c = c  # Constant for f(t) = log(t) + c*log(log(t))
        # Initialization: Play each arm once
        self.counts = [0] * n_arms  # N_a(t)
        self.est_means = [0.0] * n_arms  # μ̂_a(t)
        self.total_counts = 0

    def _kl_divergence(self, p, q):
        """
        KL divergence for Bernoulli: d(p,q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
        """
        # Handle edge cases
        p = max(min(p, 0.9999), 0.0001)  # Clamp p to avoid log(0)
        q = max(min(q, 0.9999), 0.0001)  # Clamp q to avoid log(0)
        
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    def select_arm(self):
        """
        Initialization: Play each arm once
        Algorithm Step 1: For each arm, compute U_a(t) as solution to:
                         U_a(t) = sup{q ∈ [μ̂_a, 1] : N_a * d(μ̂_a, q) ≤ f(t)}
                         where f(t) = log(t) + c*log(log(t))
        Algorithm Step 2: Select A_t ∈ argmax_a U_a(t)
        """
        # Initialization: play each arm once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # Compute f(t) = log(t) + c*log(log(t))
        if self.total_counts > 1:
            f_t = math.log(self.total_counts) + self.c * math.log(math.log(self.total_counts))
        else:
            f_t = 0
        
        # Step 1: Compute KL-UCB index for each arm using binary search
        kl_ucb_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            p = self.est_means[arm]
            threshold = f_t / self.counts[arm]
            
            # Binary search to find q such that N_a * d(p, q) = f(t)
            # i.e., find max q where d(p, q) ≤ threshold
            lower = p
            upper = 1.0
            
            # Binary search for upper confidence bound
            for _ in range(20):  # Sufficient iterations for convergence
                mid = (lower + upper) / 2.0
                if self._kl_divergence(p, mid) <= threshold:
                    lower = mid
                else:
                    upper = mid
            
            kl_ucb_values[arm] = lower
        
        # Step 2: Select arm with maximum KL-UCB value
        return kl_ucb_values.index(max(kl_ucb_values))

    def update(self, chosen_arm, reward):
        """
        Algorithm Step 3: Update N_A_t and μ̂_A_t incrementally
        """
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        old_mean = self.est_means[chosen_arm]
        self.est_means[chosen_arm] = old_mean + (1.0 / n) * (reward - old_mean)

    def give_instance_values(self):
        return {
            "est_means": self.est_means,
            "counts": self.counts,
            "c": self.c,
            "total_counts": self.total_counts
        }


class BayesUCBAgent(Agent):
    """
    Bayes-UCB Algorithm (Bernoulli Bandits)
    Reference: Algorithm box 9 in document
    """
    def __init__(self, n_arms, alpha_0=1, beta_0=1, c=0.0):
        super().__init__(n_arms)
        self.alpha_0 = alpha_0  # Prior parameter (usually 1)
        self.beta_0 = beta_0  # Prior parameter (usually 1)
        self.c = c  # Constant for quantile (usually 0)
        # Initialization: S_a(0) = 0, F_a(0) = 0
        self.successes = [0] * n_arms  # S_a(t) - number of successes
        self.failures = [0] * n_arms  # F_a(t) - number of failures
        self.total_counts = 0

    def select_arm(self):
        """
        Algorithm Step 1: For each arm, posterior is Beta(α_0 + S_a, β_0 + F_a)
        Algorithm Step 2: Let q_t = 1 - 1/(t*(log(t))^c)
        Algorithm Step 3: Compute U_a(t) = Q_{q_t}(Beta(α_0 + S_a, β_0 + F_a))
        Algorithm Step 4: Select A_t ∈ argmax_a U_a(t)
        """
        self.total_counts += 1
        
        # Step 2: Compute quantile level q_t = 1 - 1/(t*(log(t))^c)
        if self.total_counts > 1:
            if self.c > 0:
                q_t = 1.0 - 1.0 / (self.total_counts * (math.log(self.total_counts) ** self.c))
            else:
                q_t = 1.0 - 1.0 / self.total_counts
        else:
            q_t = 0.5  # Default for first round
        
        q_t = max(0.01, min(0.99, q_t))  # Clamp to avoid edge cases
        
        # Step 3: Compute q_t-quantile of posterior Beta distribution for each arm
        ucb_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            # Posterior: Beta(α_0 + S_a, β_0 + F_a)
            alpha = self.alpha_0 + self.successes[arm]
            beta = self.beta_0 + self.failures[arm]
            
            # Compute q_t-quantile of Beta(alpha, beta)
            ucb_values[arm] = stats.beta.ppf(q_t, alpha, beta)
        
        # Step 4: Select arm with maximum quantile
        return ucb_values.index(max(ucb_values))

    def update(self, chosen_arm, reward):
        """
        Algorithm Step 5: Observe X_t ∈ {0,1} and update
        S_A_t(t) = S_A_t(t-1) + X_t
        F_A_t(t) = F_A_t(t-1) + (1 - X_t)
        """
        if reward >= 0.5:  # Treat as success
            self.successes[chosen_arm] += 1
        else:  # Treat as failure
            self.failures[chosen_arm] += 1

    def give_instance_values(self):
        return {
            "est_means": [
                (self.alpha_0 + self.successes[arm]) / (self.alpha_0 + self.beta_0 + self.successes[arm] + self.failures[arm])
                for arm in range(self.n_arms)
            ],
            "successes": self.successes,
            "failures": self.failures,
            "alpha_0": self.alpha_0,
            "beta_0": self.beta_0,
            "total_counts": self.total_counts
        }


class ThompsonSamplingAgent(Agent):
    """
    Thompson Sampling (Bernoulli Bandits)
    Reference: Algorithm box 10 in document
    """
    def __init__(self, n_arms, alpha_0=1, beta_0=1):
        super().__init__(n_arms)
        self.alpha_0 = alpha_0  # Prior parameter (usually 1)
        self.beta_0 = beta_0  # Prior parameter (usually 1)
        # Initialization: S_a(0) = 0, F_a(0) = 0
        self.successes = [0] * n_arms  # S_a(t)
        self.failures = [0] * n_arms  # F_a(t)

    def select_arm(self):
        """
        Algorithm Step 1: For each arm, form posterior Beta(α_0 + S_a, β_0 + F_a)
        Algorithm Step 2: Sample θ̃_a(t) ~ Beta(α_0 + S_a, β_0 + F_a)
        Algorithm Step 3: Select A_t ∈ argmax_a θ̃_a(t)
        """
        sampled_theta = [0.0] * self.n_arms
        
        for arm in range(self.n_arms):
            # Step 1 & 2: Sample from posterior Beta(α_0 + S_a, β_0 + F_a)
            alpha = self.alpha_0 + self.successes[arm]
            beta = self.beta_0 + self.failures[arm]
            sampled_theta[arm] = random.betavariate(alpha, beta)
        
        # Step 3: Select arm with maximum sampled value
        return sampled_theta.index(max(sampled_theta))

    def update(self, chosen_arm, reward):
        """
        Algorithm Step 4: Observe X_t ∈ {0,1} and update
        S_A_t(t) = S_A_t(t-1) + X_t
        F_A_t(t) = F_A_t(t-1) + (1 - X_t)
        """
        if reward >= 0.5:  # Treat as success (for Bernoulli: reward = 1)
            self.successes[chosen_arm] += 1
        else:  # Treat as failure (for Bernoulli: reward = 0)
            self.failures[chosen_arm] += 1

    def give_instance_values(self):
        return {
            "est_means": [
                (self.alpha_0 + self.successes[arm]) / (self.alpha_0 + self.beta_0 + self.successes[arm] + self.failures[arm])
                for arm in range(self.n_arms)
            ],
            "successes": self.successes,
            "failures": self.failures,
            "alpha_0": self.alpha_0,
            "beta_0": self.beta_0
        }