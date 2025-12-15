import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '2_contextual_bandot'))

def generate_full_counterfactuals_with_consistent_seed(dataset, env, seed):
    """
    Generate full counterfactuals with guaranteed consistent random state
    This ensures all approaches get identical results at full coverage
    """
    from contextual_bandit import generate_contextual_full_counterfactuals
    if seed is not None:
        np.random.seed(seed)
    return generate_contextual_full_counterfactuals(dataset, env, seed)

def _estimate_variance_from_data(dataset, env, n_states, n_arms):
    """
    Estimate empirical variance for each state-action pair from available data
    
    Args:
        dataset: Dataset with factual and/or counterfactual annotations
        env: Environment 
        n_states: Number of states
        n_arms: Number of arms
        
    Returns:
        dict: Variance estimates {(state, action): variance}
    """
    rewards_by_state_action = {}
    
    for data_point in dataset:
        state = data_point['state']
        factual_action = data_point['action']
        factual_reward = data_point['reward']
        
        if (state, factual_action) not in rewards_by_state_action:
            rewards_by_state_action[(state, factual_action)] = []
        rewards_by_state_action[(state, factual_action)].append(factual_reward)
        
        if 'counterfactuals' in data_point:
            counterfactuals = data_point['counterfactuals']
            for action in range(n_arms):
                if action != factual_action and counterfactuals[action] is not None:
                    if (state, action) not in rewards_by_state_action:
                        rewards_by_state_action[(state, action)] = []
                    rewards_by_state_action[(state, action)].append(counterfactuals[action])
    
    variance_estimates = {}
    global_variance = None
    
    all_variances = []
    for (state, action), rewards in rewards_by_state_action.items():
        if len(rewards) > 1:
            var = np.var(rewards, ddof=1)
            variance_estimates[(state, action)] = var
            all_variances.append(var)
    
    if all_variances:
        global_variance = np.mean(all_variances)
    else:
        global_variance = np.mean([np.mean([env.stds[s][a]**2 for a in range(n_arms)]) 
                                 for s in range(n_states)])
    
    for state in range(n_states):
        for action in range(n_arms):
            if (state, action) not in variance_estimates:
                if len(rewards_by_state_action.get((state, action), [])) == 1:
                    variance_estimates[(state, action)] = global_variance
                else:
                    variance_estimates[(state, action)] = global_variance
    
    return variance_estimates

def _create_initial_annotations(dataset, env, init_budget, seed=None):
    """
    Create initial uniform random annotations for variance estimation
    
    Args:
        dataset: Original dataset
        env: Environment
        init_budget: Number of initial annotations
        seed: Random seed
        
    Returns:
        Augmented dataset with initial annotations
    """
    if seed is not None:
        np.random.seed(seed)
    
    if init_budget == 0:
        return _create_augmented_data(dataset, env, set())
    
    annotation_candidates = []
    for i, data_point in enumerate(dataset):
        factual_action = data_point['action']
        for action in range(env.n_arms):
            if action != factual_action:
                annotation_candidates.append((i, action))
    
    total_possible = len(annotation_candidates)
    if init_budget >= total_possible:
        selected_annotations = set(annotation_candidates)
    else:
        selected_indices = np.random.choice(
            total_possible, 
            size=min(init_budget, total_possible), 
            replace=False
        )
        selected_annotations = set(annotation_candidates[idx] for idx in selected_indices)
    
    return _create_augmented_data(dataset, env, selected_annotations)

def generate_divergence_prioritized_counterfactuals(dataset, env, eval_policy, behavior_policy, budget, batch_size=None, seed=None):
    """
    Policy-Divergence Prioritization (PDP) - prioritize actions with highest |pi_e(a|s) - pi_b(a|s)|
    
    Args:
        dataset (list): Original dataset
        env: The contextual bandit environment
        eval_policy (dict): Evaluation policy {state: [prob_per_action]}
        behavior_policy (dict): Behavior policy {state: [prob_per_action]}
        budget (int): Total number of counterfactual annotations allowed
        batch_size (int): Batch size for active selection (None = single batch)
        seed (int): Random seed
        
    Returns:
        list: Dataset with divergence-prioritized counterfactual annotations
    """
    if seed is not None:
        np.random.seed(seed)
    
    if budget == 0:
        return _create_augmented_data(dataset, env, set())
    
    total_possible = len(dataset) * (env.n_arms - 1)
    
    if budget >= total_possible:
        return generate_full_counterfactuals_with_consistent_seed(dataset, env, seed)
    
    if batch_size is None:
        batch_size = budget
    
    annotation_candidates = []
    for i, data_point in enumerate(dataset):
        state = data_point['state']
        factual_action = data_point['action']
        for action in range(env.n_arms):
            if action != factual_action:
                # Divergence score: |pi_e(a|s) - pi_b(a|s)|
                priority = abs(eval_policy[state][action] - behavior_policy[state][action])
                random_tiebreaker = np.random.random()
                annotation_candidates.append((i, action, priority, random_tiebreaker))
    
    annotation_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
    
    selected_annotations = set()
    for i in range(min(budget, len(annotation_candidates))):
        sample_idx, action, _, _ = annotation_candidates[i]
        selected_annotations.add((sample_idx, action))
    
    return _create_augmented_data(dataset, env, selected_annotations)

def generate_variance_prioritized_counterfactuals_factual_init(dataset, env, budget, batch_size=None, seed=None):
    """
    Empirical-Variance Prioritization (EVP) with factual-only initialization
    
    Args:
        dataset (list): Original dataset
        env: The contextual bandit environment
        budget (int): Total number of counterfactual annotations allowed
        batch_size (int): Batch size for active selection (None = single batch)
        seed (int): Random seed
        
    Returns:
        list: Dataset with variance-prioritized counterfactual annotations
    """
    if seed is not None:
        np.random.seed(seed)
    
    if budget == 0:
        return _create_augmented_data(dataset, env, set())
    
    total_possible = len(dataset) * (env.n_arms - 1)
    
    if budget >= total_possible:
        return generate_full_counterfactuals_with_consistent_seed(dataset, env, seed)
    
    current_data = _create_augmented_data(dataset, env, set())
    
    if batch_size is None:
        batch_size = budget
    
    selected_annotations = set()
    remaining_budget = budget
    
    while remaining_budget > 0:
        # Estimate variance from current data
        variance_estimates = _estimate_variance_from_data(current_data, env, env.n_states, env.n_arms)
        
        # Score all remaining candidates
        annotation_candidates = []
        for i, data_point in enumerate(dataset):
            state = data_point['state']
            factual_action = data_point['action']
            for action in range(env.n_arms):
                if action != factual_action and (i, action) not in selected_annotations:
                    priority = variance_estimates.get((state, action), 0.0)
                    random_tiebreaker = np.random.random()
                    annotation_candidates.append((i, action, priority, random_tiebreaker))
        
        if not annotation_candidates:
            break
        
        # Sort by priority (descending) and select batch
        annotation_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
        
        current_batch_size = min(batch_size, remaining_budget, len(annotation_candidates))
        batch_selections = set()
        
        for i in range(current_batch_size):
            sample_idx, action, _, _ = annotation_candidates[i]
            batch_selections.add((sample_idx, action))
            selected_annotations.add((sample_idx, action))
        
        current_data = _create_augmented_data(dataset, env, selected_annotations)
        remaining_budget -= current_batch_size
    
    return _create_augmented_data(dataset, env, selected_annotations)

def generate_variance_prioritized_counterfactuals_uniform_init(dataset, env, budget, batch_size=None, seed=None):
    """
    Empirical-Variance Prioritization (EVP) with 10% uniform initialization
    
    Args:
        dataset (list): Original dataset
        env: The contextual bandit environment
        budget (int): Total number of counterfactual annotations allowed
        batch_size (int): Batch size for active selection (None = single batch)
        seed (int): Random seed
        
    Returns:
        list: Dataset with variance-prioritized counterfactual annotations (10% uniform init)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if budget == 0:
        return _create_augmented_data(dataset, env, set())
    
    total_possible = len(dataset) * (env.n_arms - 1)
    
    if budget >= total_possible:
        return generate_full_counterfactuals_with_consistent_seed(dataset, env, seed)
    
    init_budget = max(1, int(0.1 * budget))
    remaining_budget = budget - init_budget
    
    current_data = _create_initial_annotations(dataset, env, init_budget, seed)
    
    selected_annotations = set()
    for i, data_point in enumerate(current_data):
        factual_action = data_point['action']
        if 'counterfactuals' in data_point:
            counterfactuals = data_point['counterfactuals']
            for action in range(env.n_arms):
                if action != factual_action and counterfactuals[action] is not None:
                    selected_annotations.add((i, action))
    
    if batch_size is None:
        batch_size = remaining_budget
    
    while remaining_budget > 0:
        # Estimate variance from current data
        variance_estimates = _estimate_variance_from_data(current_data, env, env.n_states, env.n_arms)
        
        # Score all remaining candidates
        annotation_candidates = []
        for i, data_point in enumerate(dataset):
            state = data_point['state']
            factual_action = data_point['action']
            for action in range(env.n_arms):
                if action != factual_action and (i, action) not in selected_annotations:
                    priority = variance_estimates.get((state, action), 0.0)
                    random_tiebreaker = np.random.random()
                    annotation_candidates.append((i, action, priority, random_tiebreaker))
        
        if not annotation_candidates:
            break
        
        # Sort by priority (descending) and select batch
        annotation_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
        
        current_batch_size = min(batch_size, remaining_budget, len(annotation_candidates))
        batch_selections = set()
        
        for i in range(current_batch_size):
            sample_idx, action, _, _ = annotation_candidates[i]
            batch_selections.add((sample_idx, action))
            selected_annotations.add((sample_idx, action))
        
        current_data = _create_augmented_data(dataset, env, selected_annotations)
        remaining_budget -= current_batch_size
    
    return _create_augmented_data(dataset, env, selected_annotations)

def generate_hybrid_divergence_variance_counterfactuals(dataset, env, eval_policy, behavior_policy, budget, batch_size=None, seed=None):
    """
    Hybrid Two-Stage Prioritization (DIEVP) - first half budget using divergence, second half using variance
    
    Args:
        dataset (list): Original dataset
        env: The contextual bandit environment
        eval_policy (dict): Evaluation policy {state: [prob_per_action]}
        behavior_policy (dict): Behavior policy {state: [prob_per_action]}
        budget (int): Total number of counterfactual annotations allowed
        batch_size (int): Batch size for active selection (None = single batch)
        seed (int): Random seed
        
    Returns:
        list: Dataset with hybrid-prioritized counterfactual annotations
    """
    if seed is not None:
        np.random.seed(seed)
    
    if budget == 0:
        return _create_augmented_data(dataset, env, set())
    
    total_possible = len(dataset) * (env.n_arms - 1)
    
    if budget >= total_possible:
        return generate_full_counterfactuals_with_consistent_seed(dataset, env, seed)
    
    stage1_budget = budget // 2
    stage2_budget = budget - stage1_budget
    
    stage1_candidates = []
    for i, data_point in enumerate(dataset):
        state = data_point['state']
        factual_action = data_point['action']
        for action in range(env.n_arms):
            if action != factual_action:
                # Divergence score: |pi_e(a|s) - pi_b(a|s)|
                priority = abs(eval_policy[state][action] - behavior_policy[state][action])
                random_tiebreaker = np.random.random()
                stage1_candidates.append((i, action, priority, random_tiebreaker))
    
    stage1_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
    
    selected_annotations = set()
    for i in range(min(stage1_budget, len(stage1_candidates))):
        sample_idx, action, _, _ = stage1_candidates[i]
        selected_annotations.add((sample_idx, action))
    
    current_data = _create_augmented_data(dataset, env, selected_annotations)
    
    if batch_size is None:
        batch_size = stage2_budget
    
    remaining_budget = stage2_budget
    
    while remaining_budget > 0:
        variance_estimates = _estimate_variance_from_data(current_data, env, env.n_states, env.n_arms)
        
        annotation_candidates = []
        for i, data_point in enumerate(dataset):
            state = data_point['state']
            factual_action = data_point['action']
            for action in range(env.n_arms):
                if action != factual_action and (i, action) not in selected_annotations:
                    priority = variance_estimates.get((state, action), 0.0)
                    random_tiebreaker = np.random.random()
                    annotation_candidates.append((i, action, priority, random_tiebreaker))
        
        if not annotation_candidates:
            break
        
        annotation_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
        
        current_batch_size = min(batch_size, remaining_budget, len(annotation_candidates))
        
        for i in range(current_batch_size):
            sample_idx, action, _, _ = annotation_candidates[i]
            selected_annotations.add((sample_idx, action))
        
        current_data = _create_augmented_data(dataset, env, selected_annotations)
        remaining_budget -= current_batch_size
    
    return _create_augmented_data(dataset, env, selected_annotations)

def _create_augmented_data(dataset, env, selected_annotations):
    """
    Helper function to create augmented dataset with selected annotations
    
    Args:
        dataset (list): Original dataset
        env: The contextual bandit environment
        selected_annotations (set): Set of (sample_idx, action) tuples to annotate
        
    Returns:
        list: Augmented dataset
    """
    augmented_data = []
    for i, data_point in enumerate(dataset):
        state = data_point['state']
        factual_action = data_point['action']
        factual_reward = data_point['reward']
        
        counterfactuals = {}
        for action in range(env.n_arms):
            if action != factual_action:
                if (i, action) in selected_annotations:
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