import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from contextual_bandit import (
    ContextualBandit, generate_contextual_dataset, generate_contextual_full_counterfactuals,
    generate_contextual_budget_limited_counterfactuals, impute_missing_contextual_counterfactuals
)

def estimate_contextual_cstar_is(augmented_data, eval_policy, n_states, n_arms):
    """
    Compute C*-IS estimate per Definition 3 of the paper (equal weights version)
    
    This is the recommended approach from the paper.
    
    Formula: v̂^C*-IS = π_e(a|s)r + Σ_{ã∈A\{a}} π_e(ã|s)g^ã
    
    With equal weights (1/|A|), π_b+ becomes uniform distribution.
    
    Args:
        augmented_data (list): Data with counterfactual annotations
        eval_policy (dict): Evaluation policy
        n_states (int): Number of states
        n_arms (int): Number of arms
        
    Returns:
        float: Policy value estimate
    """
    augmented_data = impute_missing_contextual_counterfactuals(augmented_data, n_states, n_arms)
    
    estimate = 0.0
    n_samples = len(augmented_data)
    
    for row in augmented_data:
        state = row['state']
        factual_action = row['action']
        factual_reward = row['reward']
        counterfactuals = row['counterfactuals']
        
        sample_estimate = eval_policy[state][factual_action] * factual_reward
        
        for a_tilde in range(n_arms):
            if a_tilde != factual_action and counterfactuals[a_tilde] is not None:
                sample_estimate += eval_policy[state][a_tilde] * counterfactuals[a_tilde]
        
        estimate += sample_estimate
    
    return estimate / n_samples

def estimate_contextual_cis(augmented_data, behavior_policy, eval_policy, n_states, n_arms, 
                            use_equal_weights=True):
    """
    Compute C-IS estimate per Definition 2 of the paper (general weighted version)
    
    Formula: v̂^C-IS = w^a ρ^a r + Σ_{ã∈A\{a}} w^ã ρ^ã g^ã
    where ρ^ã = π_e(ã|s) / π_b+(ã|s)
    and π_b+(a|s) = W̄(a|s,a)π_b(a|s) + Σ_{ǎ∈A\{a}} W̄(a|s,ǎ)π_b(ǎ|s)
    
    Args:
        augmented_data (list): Data with counterfactual annotations
        behavior_policy (dict): Behavior policy
        eval_policy (dict): Evaluation policy
        n_states (int): Number of states
        n_arms (int): Number of arms
        use_equal_weights (bool): If True, use equal weights (recommended)
        
    Returns:
        float: Policy value estimate
    """
    augmented_data = impute_missing_contextual_counterfactuals(augmented_data, n_states, n_arms)
    
    if use_equal_weights:
        weights = {i: {a: 1.0/n_arms for a in range(n_arms)} 
                  for i in range(len(augmented_data))}
    else:
        weights = {i: {a: 1.0/n_arms for a in range(n_arms)} 
                  for i in range(len(augmented_data))}
    
    w_bar = np.zeros((n_states, n_arms, n_arms))
    
    counts = np.zeros((n_states, n_arms))
    for i, row in enumerate(augmented_data):
        state = row['state']
        factual_action = row['action']
        counts[state, factual_action] += 1
        
        for a in range(n_arms):
            w_bar[state, factual_action, a] += weights[i][a]
    
    for state in range(n_states):
        for factual_action in range(n_arms):
            if counts[state, factual_action] > 0:
                w_bar[state, factual_action, :] /= counts[state, factual_action]
    
    pi_b_plus = np.zeros((n_states, n_arms))
    for state in range(n_states):
        for a in range(n_arms):
            for factual_action in range(n_arms):
                pi_b_plus[state, a] += (w_bar[state, factual_action, a] * 
                                       behavior_policy[state][factual_action])
    
    estimate = 0.0
    for i, row in enumerate(augmented_data):
        state = row['state']
        factual_action = row['action']
        factual_reward = row['reward']
        counterfactuals = row['counterfactuals']
        
        w = weights[i]
        
        sample_estimate = 0.0
        
        if pi_b_plus[state, factual_action] > 0:
            rho_a = eval_policy[state][factual_action] / pi_b_plus[state, factual_action]
            sample_estimate += w[factual_action] * rho_a * factual_reward
        
        for a_tilde in range(n_arms):
            if a_tilde != factual_action and counterfactuals[a_tilde] is not None:
                if pi_b_plus[state, a_tilde] > 0:
                    rho_tilde = eval_policy[state][a_tilde] / pi_b_plus[state, a_tilde]
                    sample_estimate += w[a_tilde] * rho_tilde * counterfactuals[a_tilde]
        
        estimate += sample_estimate
    
    return estimate / len(augmented_data)

def estimate_standard_is(dataset, behavior_policy, eval_policy):
    """
    Compute standard importance sampling estimate (no counterfactual annotations)

    Formula: v̂^IS = ρ r where ρ = π_e(a|s) / π_b(a|s)

    Args:
        dataset (list): Original dataset (no counterfactuals)
        behavior_policy (dict): Behavior policy
        eval_policy (dict): Evaluation policy

    Returns:
        float: Policy value estimate
    """
    estimate = 0.0

    for row in dataset:
        state = row['state']
        action = row['action']
        reward = row['reward']

        if behavior_policy[state][action] > 0:
            rho = eval_policy[state][action] / behavior_policy[state][action]
            estimate += rho * reward

    return estimate / len(dataset)

def estimate_standard_is_with_imputation(dataset, eval_policy, n_states, n_arms):
    """
    Compute IS-style estimate that uses imputed counterfactuals
    This creates a "direct method" estimate using imputed values for all actions

    Args:
        dataset (list): Original dataset (no counterfactuals)
        eval_policy (dict): Evaluation policy
        n_states (int): Number of states
        n_arms (int): Number of arms

    Returns:
        float: Policy value estimate using imputed counterfactuals
    """
    augmented_data = []
    for row in dataset:
        state = row['state']
        factual_action = row['action']
        factual_reward = row['reward']

        counterfactuals = {}
        for action in range(n_arms):
            if action != factual_action:
                counterfactuals[action] = None

        augmented_data.append({
            'state': state,
            'action': factual_action,
            'reward': factual_reward,
            'counterfactuals': counterfactuals
        })

    augmented_data = impute_missing_contextual_counterfactuals(augmented_data, n_states, n_arms)

    estimate = 0.0
    for row in augmented_data:
        state = row['state']
        factual_action = row['action']
        factual_reward = row['reward']
        counterfactuals = row['counterfactuals']

        sample_estimate = eval_policy[state][factual_action] * factual_reward

        for action in range(n_arms):
            if action != factual_action and counterfactuals[action] is not None:
                sample_estimate += eval_policy[state][action] * counterfactuals[action]

        estimate += sample_estimate

    return estimate / len(augmented_data)

def evaluate_contextual_approaches(env, policies, policy_names, n_datasets=100, 
                                  samples_per_dataset=1000, annotation_budgets=[500, 1000, 2000, 3000]):
    """
    Evaluate and compare different OPE approaches in contextual setting
    """
    behavior_policy = {
        0: [0.4, 0.3, 0.1, 0.1, 0.1],
        1: [0.1, 0.1, 0.4, 0.3, 0.1],
        2: [0.2, 0.2, 0.2, 0.2, 0.2]   
    }

    n_states = env.n_states
    n_arms = env.n_arms
    
    true_values = [env.get_true_value(policy) for policy in policies]
    
    results = {
        "standard_is": {policy_name: {"estimates": [], "rmse": 0, "std": 0, "mean": 0}
                       for policy_name in policy_names},
        "standard_is_imputed": {policy_name: {"estimates": [], "rmse": 0, "std": 0, "mean": 0}
                               for policy_name in policy_names},
        "cstar_is_full": {policy_name: {"estimates": [], "rmse": 0, "std": 0, "mean": 0}
                         for policy_name in policy_names},
        "cis_full": {policy_name: {"estimates": [], "rmse": 0, "std": 0, "mean": 0}
                    for policy_name in policy_names}
    }
    
    for budget in annotation_budgets:
        results[f"cstar_is_budget_{budget}"] = {
            policy_name: {"estimates": [], "rmse": 0, "std": 0, "mean": 0} 
            for policy_name in policy_names
        }
        results[f"cis_budget_{budget}"] = {
            policy_name: {"estimates": [], "rmse": 0, "std": 0, "mean": 0} 
            for policy_name in policy_names
        }
    
    print("Running contextual bandit experiments...")
    
    for dataset_idx in tqdm(range(n_datasets), desc="Generating datasets"):
        seed = dataset_idx
        
        dataset = generate_contextual_dataset(env, behavior_policy, samples_per_dataset, seed)
        
        for i, (policy, policy_name) in enumerate(zip(policies, policy_names)):
            true_value = true_values[i]
            
            is_estimate = estimate_standard_is(dataset, behavior_policy, policy)
            results["standard_is"][policy_name]["estimates"].append(is_estimate)

            is_imputed_estimate = estimate_standard_is_with_imputation(dataset, policy, n_states, n_arms)
            results["standard_is_imputed"][policy_name]["estimates"].append(is_imputed_estimate)
            
            # Apply same multiple trial averaging to full annotations for fair comparison
            n_annotation_trials = 10
            cstar_full_estimates = []
            cis_full_estimates = []
            
            for trial in range(n_annotation_trials):
                trial_seed = seed + trial * 10000
                trial_full_cf = generate_contextual_full_counterfactuals(dataset, env, trial_seed)
                
                cstar_estimate = estimate_contextual_cstar_is(
                    trial_full_cf, policy, env.n_states, env.n_arms)
                cstar_full_estimates.append(cstar_estimate)
                
                cis_estimate = estimate_contextual_cis(
                    trial_full_cf, behavior_policy, policy, env.n_states, env.n_arms)
                cis_full_estimates.append(cis_estimate)
            
            results["cstar_is_full"][policy_name]["estimates"].append(np.mean(cstar_full_estimates))
            results["cis_full"][policy_name]["estimates"].append(np.mean(cis_full_estimates))
            
            # Budget-limited versions - use same number of trials as full annotations
            for budget in annotation_budgets:
                cstar_estimates = []
                cis_estimates = []

                for trial in range(n_annotation_trials):
                    trial_seed = seed + trial * 10000
                    trial_budget_cf = generate_contextual_budget_limited_counterfactuals(
                        dataset, env, budget, trial_seed)

                    cstar_estimate = estimate_contextual_cstar_is(
                        trial_budget_cf, policy, env.n_states, env.n_arms)
                    cstar_estimates.append(cstar_estimate)

                    cis_estimate = estimate_contextual_cis(
                        trial_budget_cf, behavior_policy, policy, env.n_states, env.n_arms)
                    cis_estimates.append(cis_estimate)

                results[f"cstar_is_budget_{budget}"][policy_name]["estimates"].append(np.mean(cstar_estimates))
                results[f"cis_budget_{budget}"][policy_name]["estimates"].append(np.mean(cis_estimates))
    
    for approach in results:
        for policy_name in policy_names:
            estimates = results[approach][policy_name]["estimates"]
            true_value = true_values[policy_names.index(policy_name)]
            
            results[approach][policy_name]["mean"] = np.mean(estimates)
            results[approach][policy_name]["std"] = np.std(estimates)
            results[approach][policy_name]["rmse"] = np.sqrt(np.mean([(est - true_value)**2 for est in estimates]))
    
    return results, true_values

if __name__ == "__main__":
    env = ContextualBandit(n_states=3, n_arms=5, seed=42)
    
    policies = [
        {0: [1, 0, 0, 0, 0], 1: [0, 0, 1, 0, 0], 2: [0, 0, 0, 0, 1]},
        {0: [0.2, 0.2, 0.2, 0.2, 0.2], 1: [0.2, 0.2, 0.2, 0.2, 0.2], 2: [0.2, 0.2, 0.2, 0.2, 0.2]},
        {0: [0, 1, 0, 0, 0], 1: [0, 0, 0, 1, 0], 2: [0.25, 0.25, 0.25, 0.25, 0]}
    ]
    
    policy_names = ["State-Optimal", "Uniform", "Suboptimal"]
    
    results, true_values = evaluate_contextual_approaches(
        env, policies, policy_names,
        n_datasets=50,
        samples_per_dataset=1000,
        annotation_budgets=[0, 500, 1000, 1500, 2000, 3000]
    )
    
    print("\n" + "="*80)
    print("CONTEXTUAL BANDIT RESULTS (Paper-Compliant Implementation)")
    print("="*80)
    
    for i, policy_name in enumerate(policy_names):
        print(f"\n{policy_name} Policy (True Value: {true_values[i]:.3f})")
        print("-" * 80)
        
        mean_est = results["standard_is"][policy_name]["mean"]
        rmse = results["standard_is"][policy_name]["rmse"]
        print(f"{'Standard IS':30s}: Mean={mean_est:.3f}, RMSE={rmse:.3f}")

        mean_est = results["standard_is_imputed"][policy_name]["mean"]
        rmse = results["standard_is_imputed"][policy_name]["rmse"]
        print(f"{'Standard IS + Imputation':30s}: Mean={mean_est:.3f}, RMSE={rmse:.3f}")
        
        for budget in [0, 500, 1000, 1500, 2000, 3000]:
            approach = f"cstar_is_budget_{budget}"
            mean_est = results[approach][policy_name]["mean"]
            rmse = results[approach][policy_name]["rmse"]
            print(f"{f'C*-IS (budget={budget})':30s}: Mean={mean_est:.3f}, RMSE={rmse:.3f}")
        
        mean_est = results["cstar_is_full"][policy_name]["mean"]
        rmse = results["cstar_is_full"][policy_name]["rmse"]
        print(f"{'C*-IS (full annotations)':30s}: Mean={mean_est:.3f}, RMSE={rmse:.3f}")
    
    fig, axes = plt.subplots(1, len(policy_names), figsize=(15, 5))
    budgets = [0, 500, 1000, 1500, 2000, 3000]
    
    for i, policy_name in enumerate(policy_names):
        # Plot C*-IS budget-limited results
        rmse_values_cstar = []
        for budget in budgets:
            rmse_values_cstar.append(results[f"cstar_is_budget_{budget}"][policy_name]["rmse"])
        
        axes[i].plot(budgets, rmse_values_cstar, 'o-', linewidth=2, markersize=6, label='C*-IS')
        
        # Add horizontal reference lines
        axes[i].axhline(y=results["standard_is"][policy_name]["rmse"],
                       color='gray', linestyle='--', label='Standard IS')
        axes[i].axhline(y=results["standard_is_imputed"][policy_name]["rmse"],
                       color='orange', linestyle='--', label='Standard IS + Imputation')
        axes[i].axhline(y=results["cstar_is_full"][policy_name]["rmse"],
                       color='blue', linestyle=':', linewidth=2, label='C*-IS (Full Annotations)')
        
        axes[i].set_xlabel('Annotation Budget')
        axes[i].set_ylabel('RMSE')
        axes[i].set_title(f'{policy_name} Policy')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    
    # Create directories if they don't exist and save plot
    images_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'images')
    os.makedirs(images_dir, exist_ok=True)
    plot_path = os.path.join(images_dir, 'contextual_bandit_paper_compliant.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'images/contextual_bandit_paper_compliant.png'")
    
    results_df = []
    for approach in results:
        for policy_name in policy_names:
            results_df.append({
                'approach': approach,
                'policy': policy_name,
                'mean_estimate': results[approach][policy_name]["mean"],
                'rmse': results[approach][policy_name]["rmse"],
                'std': results[approach][policy_name]["std"],
                'true_value': true_values[policy_names.index(policy_name)]
            })
    
    df = pd.DataFrame(results_df)
    outputs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    csv_path = os.path.join(outputs_dir, 'contextual_bandit_paper_compliant_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Detailed results saved as 'outputs/contextual_bandit_paper_compliant_results.csv'")