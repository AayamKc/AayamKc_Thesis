import numpy as np

class ContextualBandit:
    def __init__(self, n_states=3, n_arms=5, seed=None):
        """
        Initialize a contextual bandit environment with multiple states and arms
        
        Args:
            n_states (int): Number of states/contexts
            n_arms (int): Number of arms
            seed (int): Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_arms = n_arms
        
        if seed is not None:
            np.random.seed(seed)
        
        self.means = {
            0: [0.8, 0.7, 0.3, 0.2, 0.1],  
            1: [0.2, 0.3, 0.8, 0.7, 0.2],  
            2: [0.4, 0.4, 0.4, 0.4, 1.0]   
        }
        
        self.stds = {
            0: [0.2, 0.2, 0.3, 0.3, 0.3],  
            1: [0.3, 0.3, 0.2, 0.2, 0.3],  
            2: [0.2, 0.2, 0.2, 0.2, 0.1]   
        }

    def step(self, state, action):
        """
        Take one step in the environment
        
        Args:
            state (int): Current state (0, 1, or 2)
            action (int): Which arm to pull (0-4)
            
        Returns:
            float: The reward received
        """
        assert 0 <= state < self.n_states, f"Invalid state {state}"
        assert 0 <= action < self.n_arms, f"Invalid action {action}"
        
        reward = np.random.normal(self.means[state][action], self.stds[state][action])
        return reward
    
    def get_true_value(self, policy):
        """
        Calculate true value of a policy across all states
        
        Args:
            policy (dict): Dict mapping state -> list of probabilities for each arm
            
        Returns:
            float: Expected value of the policy (assuming uniform state distribution)
        """
        total_value = 0.0
        for state in range(self.n_states):
            assert len(policy[state]) == self.n_arms, f"Policy must specify probs for all arms in state {state}"
            assert abs(sum(policy[state]) - 1.0) < 1e-6, f"Policy probabilities must sum to 1 in state {state}"
            
            state_value = sum(p * mu for p, mu in zip(policy[state], self.means[state]))
            total_value += state_value
        
        return total_value / self.n_states

def generate_contextual_dataset(env, behavior_policy, n_samples, seed=None):
    """
    Generate a dataset using the behavior policy in the contextual environment
    
    Args:
        env (ContextualBandit): The contextual bandit environment
        behavior_policy (dict): Dict mapping state -> list of action probabilities
        n_samples (int): Number of samples to generate
        seed (int): Random seed
        
    Returns:
        list: Dataset with state, action, and reward for each sample
    """
    if seed is not None:
        np.random.seed(seed)
    
    dataset = []
    for _ in range(n_samples):
        state = np.random.randint(0, env.n_states)
        action = np.random.choice(env.n_arms, p=behavior_policy[state])
        reward = env.step(state, action)
        
        dataset.append({
            'state': state,
            'action': action,
            'reward': reward
        })
    
    return dataset

def generate_contextual_full_counterfactuals(dataset, env, seed=None):
    """
    Generate full counterfactual annotations for all non-factual actions
    Per paper: annotations are for counterfactual actions only
    
    Args:
        dataset (list): Original dataset
        env (ContextualBandit): The environment
        seed (int): Random seed
        
    Returns:
        list: Dataset augmented with full counterfactual annotations
    """
    if seed is not None:
        np.random.seed(seed)
    
    augmented_data = []
    for data_point in dataset:
        state = data_point['state']
        factual_action = data_point['action']
        factual_reward = data_point['reward']
        
        counterfactuals = {}
        for action in range(env.n_arms):
            if action != factual_action:
                counterfactuals[action] = env.step(state, action)
        
        augmented_data.append({
            'state': state,
            'action': factual_action,
            'reward': factual_reward,
            'counterfactuals': counterfactuals
        })
    
    return augmented_data

def generate_contextual_budget_limited_counterfactuals(dataset, env, budget, seed=None):
    """
    Generate budget-limited counterfactual annotations using random selection
    
    Args:
        dataset (list): Original dataset
        env (ContextualBandit): The environment
        budget (int): Total number of counterfactual annotations allowed
        seed (int): Random seed
        
    Returns:
        list: Dataset with budget-limited counterfactual annotations
    """
    if seed is not None:
        np.random.seed(seed)
    
    total_possible = len(dataset) * (env.n_arms - 1)
    
    if budget >= total_possible:
        return generate_contextual_full_counterfactuals(dataset, env, seed)
    
    annotation_candidates = []
    for i, data_point in enumerate(dataset):
        factual_action = data_point['action']
        for action in range(env.n_arms):
            if action != factual_action:
                annotation_candidates.append((i, action))
    
    np.random.shuffle(annotation_candidates)
    selected_annotations = np.random.choice(
        len(annotation_candidates), 
        size=min(budget, len(annotation_candidates)), 
        replace=False, 
    )
    
    selected_set = set()
    for idx in selected_annotations:
        selected_set.add(annotation_candidates[idx])
    
    augmented_data = []
    for i, data_point in enumerate(dataset):
        state = data_point['state']
        factual_action = data_point['action']
        factual_reward = data_point['reward']
        
        counterfactuals = {}
        for action in range(env.n_arms):
            if action != factual_action:
                if (i, action) in selected_set:
                    counterfactuals[action] = env.step(state, action)
                else:
                    counterfactuals[action] = None
        
        augmented_data.append({
            'state': state,
            'action': factual_action,
            'reward': factual_reward,
            'counterfactuals': counterfactuals
        })
    
    return augmented_data

def impute_missing_contextual_counterfactuals(augmented_data, n_states, n_arms):
    """
    Impute missing counterfactual annotations using averages per state-action pair
    Following Appendix D.2 of the paper
    
    Args:
        augmented_data (list): Data with potentially missing annotations
        n_states (int): Number of states
        n_arms (int): Number of arms
        
    Returns:
        list: Data with imputed annotations
    """
    annotation_sums = {(state, arm): 0.0 for state in range(n_states) for arm in range(n_arms)}
    annotation_counts = {(state, arm): 0 for state in range(n_states) for arm in range(n_arms)}
    
    for data_point in augmented_data:
        state = data_point['state']
        
        factual_action = data_point['action']
        factual_reward = data_point['reward']
        annotation_sums[(state, factual_action)] += factual_reward
        annotation_counts[(state, factual_action)] += 1
        
        counterfactuals = data_point['counterfactuals']
        for arm in range(n_arms):
            if arm != factual_action and counterfactuals[arm] is not None:
                annotation_sums[(state, arm)] += counterfactuals[arm]
                annotation_counts[(state, arm)] += 1
    
    annotation_avgs = {}
    for state in range(n_states):
        for arm in range(n_arms):
            if annotation_counts[(state, arm)] > 0:
                annotation_avgs[(state, arm)] = annotation_sums[(state, arm)] / annotation_counts[(state, arm)]
            else:
                annotation_avgs[(state, arm)] = None
    
    imputed_data = []
    for data_point in augmented_data:
        imputed_point = data_point.copy()
        imputed_counterfactuals = data_point['counterfactuals'].copy()
        state = data_point['state']
        factual_action = data_point['action']
        
        for arm in range(n_arms):
            if arm != factual_action and imputed_counterfactuals[arm] is None:
                if annotation_avgs[(state, arm)] is not None:
                    imputed_counterfactuals[arm] = annotation_avgs[(state, arm)]
        
        imputed_point['counterfactuals'] = imputed_counterfactuals
        imputed_data.append(imputed_point)
    
    return imputed_data