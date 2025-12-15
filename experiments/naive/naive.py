import numpy as np
import sys
import os
from typing import Dict, List, Tuple

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

def generate_greedy_pie_only_counterfactuals(dataset, env, eval_policy, budget, seed=None):
    """
    Greedy πe-only - always annotate the action with highest πe(a|s)
    This is intuitive ("annotate what the evaluation policy cares about") 
    but ignores the importance weight structure
    
    Args:
        dataset (list): Original dataset
        env: The contextual bandit environment
        eval_policy (dict): Evaluation policy {state: [prob_per_action]}
        budget (int): Total number of counterfactual annotations allowed
        seed (int): Random seed
        
    Returns:
        list: Dataset with greedy πe-only counterfactual annotations
    """
    if seed is not None:
        np.random.seed(seed)
    
    if budget == 0:
        return _create_augmented_data(dataset, env, set())
    
    total_possible = len(dataset) * (env.n_arms - 1)
    
    if budget >= total_possible:
        return generate_full_counterfactuals_with_consistent_seed(dataset, env, seed)
    
    annotation_candidates = []
    for i, data_point in enumerate(dataset):
        state = data_point['state']
        factual_action = data_point['action']
        for action in range(env.n_arms):
            if action != factual_action:
                priority = eval_policy[state][action]
                random_tiebreaker = np.random.random()
                annotation_candidates.append((i, action, priority, random_tiebreaker))
    
    annotation_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
    selected_annotations = set()
    for i in range(min(budget, len(annotation_candidates))):
        sample_idx, action, _, _ = annotation_candidates[i]
        selected_annotations.add((sample_idx, action))
    
    return _create_augmented_data(dataset, env, selected_annotations)


def generate_greedy_support_based_counterfactuals(dataset, env, behavior_policy, budget, seed=None):
    """
    Greedy support-based - prioritize actions where πb(a|s) is lowest 
    (i.e., least support in data). This targets the "support deficiency" 
    problem but ignores πe entirely
    
    Args:
        dataset (list): Original dataset
        env: The contextual bandit environment  
        behavior_policy (dict): Behavior policy {state: [prob_per_action]}
        budget (int): Total number of counterfactual annotations allowed
        seed (int): Random seed
        
    Returns:
        list: Dataset with greedy support-based counterfactual annotations
    """
    if seed is not None:
        np.random.seed(seed)
    
    if budget == 0:
        return _create_augmented_data(dataset, env, set())
    
    total_possible = len(dataset) * (env.n_arms - 1)
    
    if budget >= total_possible:
        return generate_full_counterfactuals_with_consistent_seed(dataset, env, seed)
    
    annotation_candidates = []
    for i, data_point in enumerate(dataset):
        state = data_point['state']
        factual_action = data_point['action']
        for action in range(env.n_arms):
            if action != factual_action:
                priority = 1.0 / (behavior_policy[state][action] + 1e-8)
                random_tiebreaker = np.random.random()
                annotation_candidates.append((i, action, priority, random_tiebreaker))
    
    annotation_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
    selected_annotations = set()
    for i in range(min(budget, len(annotation_candidates))):
        sample_idx, action, _, _ = annotation_candidates[i]
        selected_annotations.add((sample_idx, action))
    
    return _create_augmented_data(dataset, env, selected_annotations)


def generate_round_robin_counterfactuals(dataset, env, budget, seed=None):
    """
    Round-robin by action - cycle through actions ensuring each action type 
    gets equal annotation budget. This ensures coverage but ignores state-specific needs
    
    Args:
        dataset (list): Original dataset
        env: The contextual bandit environment
        budget (int): Total number of counterfactual annotations allowed
        seed (int): Random seed
        
    Returns:
        list: Dataset with round-robin counterfactual annotations
    """
    if seed is not None:
        np.random.seed(seed)
    
    if budget == 0:
        return _create_augmented_data(dataset, env, set())
    
    total_possible = len(dataset) * (env.n_arms - 1)
    
    if budget >= total_possible:
        return generate_full_counterfactuals_with_consistent_seed(dataset, env, seed)
    
    candidates_by_action = {action: [] for action in range(env.n_arms)}
    for i, data_point in enumerate(dataset):
        factual_action = data_point['action']
        for action in range(env.n_arms):
            if action != factual_action:
                candidates_by_action[action].append((i, action))
    
    for action in range(env.n_arms):
        np.random.shuffle(candidates_by_action[action])
    
    selected_annotations = set()
    allocated_budget = 0
    action_idx = 0
    
    while allocated_budget < budget:
        attempts = 0
        while attempts < env.n_arms:
            if candidates_by_action[action_idx]:
                sample_idx, action = candidates_by_action[action_idx].pop(0)
                selected_annotations.add((sample_idx, action))
                allocated_budget += 1
                break
            action_idx = (action_idx + 1) % env.n_arms
            attempts += 1
        
        if attempts == env.n_arms:
            break
            
        action_idx = (action_idx + 1) % env.n_arms
    
    return _create_augmented_data(dataset, env, selected_annotations)


def generate_high_reward_first_counterfactuals(dataset, env, budget, seed=None):
    """
    High-reward-first - annotate actions you expect to have high rewards.
    This is what a naive practitioner might do but conflates reward magnitude 
    with estimation importance. Uses environment's true means as prior.
    
    Args:
        dataset (list): Original dataset
        env: The contextual bandit environment
        budget (int): Total number of counterfactual annotations allowed
        seed (int): Random seed
        
    Returns:
        list: Dataset with high-reward-first counterfactual annotations
    """
    if seed is not None:
        np.random.seed(seed)
    
    if budget == 0:
        return _create_augmented_data(dataset, env, set())
    
    total_possible = len(dataset) * (env.n_arms - 1)
    
    if budget >= total_possible:
        return generate_full_counterfactuals_with_consistent_seed(dataset, env, seed)
    
    annotation_candidates = []
    for i, data_point in enumerate(dataset):
        state = data_point['state']
        factual_action = data_point['action']
        for action in range(env.n_arms):
            if action != factual_action:
                expected_reward = env.means[state][action]
                random_tiebreaker = np.random.random()
                annotation_candidates.append((i, action, expected_reward, random_tiebreaker))
    
    annotation_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
    selected_annotations = set()
    for i in range(min(budget, len(annotation_candidates))):
        sample_idx, action, _, _ = annotation_candidates[i]
        selected_annotations.add((sample_idx, action))
    
    return _create_augmented_data(dataset, env, selected_annotations)


def generate_uniform_random_counterfactuals(dataset, env, budget, seed=None):
    """
    Uniform Random - randomly select counterfactuals to annotate
    This is the baseline approach (already implemented as generate_contextual_budget_limited_counterfactuals)
    but included here for completeness and consistency
    
    Args:
        dataset (list): Original dataset
        env: The contextual bandit environment
        budget (int): Total number of counterfactual annotations allowed
        seed (int): Random seed
        
    Returns:
        list: Dataset with uniform random counterfactual annotations
    """
    if seed is not None:
        np.random.seed(seed)
    
    if budget == 0:
        return _create_augmented_data(dataset, env, set())
    
    total_possible = len(dataset) * (env.n_arms - 1)
    
    if budget >= total_possible:
        return generate_full_counterfactuals_with_consistent_seed(dataset, env, seed)
    
    from contextual_bandit import generate_contextual_budget_limited_counterfactuals
    return generate_contextual_budget_limited_counterfactuals(dataset, env, budget, seed)


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