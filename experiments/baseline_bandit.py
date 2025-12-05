import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars


class FiveArmedBandit:
    def __init__(self, means=None, stds=None, seed=None):
        """
        Initialize a 5-armed bandit environment
        
        Args:
            means (list): Mean rewards for each arm. Defaults to [0.2, 0.5, 0.8, 1.0, 1.5]
            stds (list): Standard deviations for each arm. Defaults to [0.1, 0.2, 0.3, 0.4, 0.5]
        """
        self.n_arms = 5
        self.means = means if means is not None else [0.2, 0.5, 0.8, 1.0, 1.5]
        self.stds = stds if stds is not None else [0.1, 0.2, 0.3, 0.4, 0.5]
        
        if seed is not None:
            np.random.seed(seed)

    def step(self, action):
        """
        Take one step in the environment by pulling an arm
        
        Args:
            action (int): Which arm to pull (0-4)
            
        Returns:
            float: The reward received
        """
        assert 0 <= action < self.n_arms, f"Invalid action {action}"
        reward = np.random.normal(self.means[action], self.stds[action])
        return reward
    
    def get_true_value(self, policy):
        """
        Calculate true value of a policy
        
        Args:
            policy (list): List of probabilities for each arm
            
        Returns:
            float: Expected value of the policy
        """
        assert len(policy) == self.n_arms, "Policy must specify probs for all arms"
        assert abs(sum(policy) - 1.0) < 1e-6, "Policy probabilities must sum to 1"
        
        return sum(p * mu for p, mu in zip(policy, self.means))
    
def generate_dataset(env, policy, n_samples):
    """
    Generate a dataset of (action, reward) pairs using given policy
    
    Args:
        env (FiveArmedBandit): The bandit environment
        policy (list): List of action probabilities
        n_samples (int): Number of samples to generate
        
    Returns:
        list: List of (action, reward) tuples
    """
    dataset = []
    for _ in range(n_samples):
        action = np.random.choice(env.n_arms, p=policy)
        reward = env.step(action)
        dataset.append((action, reward))
    return dataset

def generate_full_counterfactuals(env, dataset):
    """
    Generate counterfactual rewards for all arms by sampling from the reward distribution
    
    Args:
        env (FiveArmedBandit): The bandit environment
        dataset: List of (action, reward) from original data collection
        
    Returns:
        list: Original data augmented with counterfactual rewards
    """
    augmented_data = []
    
    for action, reward in dataset:
        counterfactual_rewards = []
        for arm in range(env.n_arms):
            if arm == action:
                counterfactual_rewards.append(reward)
            else:
                sampled_reward = np.random.normal(env.means[arm], env.stds[arm])
                counterfactual_rewards.append(sampled_reward)
        
        augmented_data.append({
            'action': action,
            'reward': reward,
            'counterfactuals': counterfactual_rewards
        })
    
    return augmented_data

def impute_missing_counterfactuals(augmented_data, n_arms):
    """
    Impute missing counterfactual annotations using averages of available annotations.
    
    Args:
        augmented_data: List of dictionaries with counterfactual annotations
        n_arms: Number of arms in the bandit
        
    Returns:
        List of dictionaries with imputed counterfactual annotations
    """
    annotation_sums = {arm: 0.0 for arm in range(n_arms)}
    annotation_counts = {arm: 0 for arm in range(n_arms)}
    
    for data_point in augmented_data:
        action = data_point['action']
        counterfactuals = data_point['counterfactuals']
        
        for arm in range(n_arms):
            if arm != action and counterfactuals[arm] is not None:
                annotation_sums[arm] += counterfactuals[arm]
                annotation_counts[arm] += 1
    
    annotation_avgs = {}
    for arm in range(n_arms):
        if annotation_counts[arm] > 0:
            annotation_avgs[arm] = annotation_sums[arm] / annotation_counts[arm]
        else:
            annotation_avgs[arm] = None  

    imputed_data = []
    for data_point in augmented_data:
        imputed_point = data_point.copy()
        imputed_counterfactuals = data_point['counterfactuals'].copy()
        
        for arm in range(n_arms):
            if arm != data_point['action'] and imputed_counterfactuals[arm] is None and annotation_avgs[arm] is not None:
                imputed_counterfactuals[arm] = annotation_avgs[arm]
        
        imputed_point['counterfactuals'] = imputed_counterfactuals
        imputed_data.append(imputed_point)
    
    return imputed_data



def generate_budget_limited_counterfactuals(env, dataset, annotation_budget):
    """
    Generate counterfactual rewards with a fixed annotation budget
    
    Args:
        env (FiveArmedBandit): The bandit environment
        dataset: List of (action, reward) from original data collection
        annotation_budget: Total number of counterfactual annotations to generate
        
    Returns:
        list: Original data augmented with budget-limited counterfactual rewards
    """
    augmented_data = []
    n_samples = len(dataset)
    
    
    for action, reward in dataset:
        counterfactual_rewards = [None] * env.n_arms
        counterfactual_rewards[action] = reward  
        
        augmented_data.append({
            'action': action,
            'reward': reward,
            'counterfactuals': counterfactual_rewards
        })
    
   
    possible_annotations = []
    for sample_idx in range(n_samples):
        sample_action = dataset[sample_idx][0]
        for arm in range(env.n_arms):
            if arm != sample_action:  
                possible_annotations.append((sample_idx, arm))
    

    np.random.shuffle(possible_annotations)
    selected_annotations = possible_annotations[:annotation_budget]
    

    for sample_idx, arm in selected_annotations:
        sampled_reward = np.random.normal(env.means[arm], env.stds[arm])
        augmented_data[sample_idx]['counterfactuals'][arm] = sampled_reward
    
    return augmented_data

def estimate_full_cis(augmented_data, behavior_policy, eval_policy):
    """
    Compute C*-IS estimate (equal weights) using fully augmented dataset.
    
    Args:
        augmented_data (list): Output from generate_full_counterfactuals
        behavior_policy (list): Original behavior policy probabilities
        eval_policy (list): Evaluation policy probabilities
    
    Returns:
        float: Estimated policy value using C*-IS
    """
    n_arms = len(behavior_policy)
    weight = 1.0 / n_arms
    
    augmented_behavior = [1.0/n_arms] * n_arms
    
    estimate = 0
    n_samples = len(augmented_data)
    
    for row in augmented_data:
        factual_action = row['action']
        factual_reward = row['reward']
        counterfactuals = row['counterfactuals']
        
    
        rho_a = eval_policy[factual_action] / augmented_behavior[factual_action]
        estimate += weight * rho_a * factual_reward
        
        
        for a in range(n_arms):
            if a != factual_action:
                rho = eval_policy[a] / augmented_behavior[a]
                estimate += weight * rho * counterfactuals[a]
    
    return estimate / n_samples


def estimate_budget_limited_cis(augmented_data, behavior_policy, eval_policy):
    """
    Compute C-IS estimate with budget-limited annotations.
    This version correctly reweights each sample based on
    which arms are available (factual + any annotated).

    Args:
        augmented_data (list): Output from generate_budget_limited_counterfactuals
        behavior_policy (list): Original behavior policy probabilities
        eval_policy (list): Evaluation policy probabilities

    Returns:
        float: Estimated policy value using budget-limited C-IS
    """
    n_arms = len(behavior_policy)
    n_samples = len(augmented_data)
    estimate = 0.0

    weights = [[] for _ in range(n_arms)]

    for i, row in enumerate(augmented_data):
        factual_action = row['action']
        counterfactuals = row['counterfactuals']
        available_arms = [factual_action] + [a for a in range(n_arms) if counterfactuals[a] is not None]

        w = np.zeros(n_arms)
        w[available_arms] = 1
        w = w / len(available_arms)
        weights[factual_action].append(w)

    w_bar = np.zeros((n_arms, n_arms))
    for factual_action in range(n_arms):
        w_a = np.asarray(weights[factual_action])
        w_a_bar = w_a.mean(axis=0)
        w_bar[factual_action] = w_a_bar

    pi_b_plus = np.zeros(n_arms)
    for a in range(n_arms):
        pi_b_plus[a] = np.sum([w_bar[factual_action][a] * behavior_policy[factual_action] 
                               for factual_action in range(n_arms)])
    
    estimate = 0.0
    for row in augmented_data:
        factual_action = row['action']
        reward = row['reward']
        counterfactuals = row['counterfactuals']
        
        w = w_bar[factual_action]
        
        sample_estimate = 0.0     

        if pi_b_plus[factual_action] > 0:
            rho_a = eval_policy[factual_action] / pi_b_plus[factual_action]
            sample_estimate += w[factual_action] * rho_a * reward
        
        
        for a_hat in range(n_arms):
            if a_hat != factual_action and counterfactuals[a_hat] is not None:
                if pi_b_plus[a_hat] > 0: 
                    rho_a_hat = eval_policy[a_hat] / pi_b_plus[a_hat]
                    sample_estimate += w[a_hat] * rho_a_hat * counterfactuals[a_hat]
        
        estimate += sample_estimate
    

    return estimate / n_samples