import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '2_contextual_bandot'))

from contextual_bandit import (
    ContextualBandit, generate_contextual_dataset, 
    impute_missing_contextual_counterfactuals
)

from contextual_bandit_runner import (
    estimate_contextual_cstar_is, estimate_contextual_cis,
    estimate_standard_is, estimate_standard_is_with_imputation
)

from active import (
    generate_divergence_prioritized_counterfactuals,
    generate_variance_prioritized_counterfactuals_factual_init,
    generate_variance_prioritized_counterfactuals_uniform_init,
    generate_hybrid_divergence_variance_counterfactuals,
    generate_full_counterfactuals_with_consistent_seed
)

def evaluate_active_approaches(env, policies, policy_names, n_datasets=100, 
                              samples_per_dataset=1000, annotation_budgets=[500, 1000, 2000, 3000],
                              batch_sizes=[None], n_annotation_trials=10):
    """
    Evaluate and compare different active learning approaches for OPE
    
    Args:
        env: ContextualBandit environment
        policies: List of evaluation policies
        policy_names: Names for the policies
        n_datasets: Number of datasets to generate
        samples_per_dataset: Number of samples per dataset
        annotation_budgets: List of annotation budgets to test
        batch_sizes: List of batch sizes to test (None = single batch)
        n_annotation_trials: Number of annotation trials per dataset
    """
    behavior_policy = {
        0: [0.4, 0.3, 0.1, 0.1, 0.1],
        1: [0.1, 0.1, 0.4, 0.3, 0.1],
        2: [0.2, 0.2, 0.2, 0.2, 0.2]   
    }

    n_states = env.n_states
    n_arms = env.n_arms
    
    true_values = [env.get_true_value(policy) for policy in policies]
    
    results = {}
    
    baseline_approaches = [
        "standard_is",
        "standard_is_imputed",
        "cstar_is_full",
        "uniform_random"
    ]
    
    for approach in baseline_approaches:
        results[approach] = {policy_name: {"estimates": [], "rmse": 0, "std": 0, "mean": 0}
                            for policy_name in policy_names}
    
    active_approaches = [
        "divergence_prioritized",
        "variance_prioritized_factual_init",
        "variance_prioritized_uniform_init", 
        "hybrid_divergence_variance"
    ]
    
    for approach in active_approaches:
        for budget in annotation_budgets:
            for batch_size in batch_sizes:
                batch_str = "single" if batch_size is None else str(batch_size)
                key = f"{approach}_budget_{budget}_batch_{batch_str}"
                results[key] = {policy_name: {"estimates": [], "rmse": 0, "std": 0, "mean": 0}
                               for policy_name in policy_names}
    
    print("Running active learning experiments...")
    
    for dataset_idx in tqdm(range(n_datasets), desc="Generating datasets"):
        seed = dataset_idx
        
        dataset = generate_contextual_dataset(env, behavior_policy, samples_per_dataset, seed)
        
        for i, (policy, policy_name) in enumerate(zip(policies, policy_names)):
            true_value = true_values[i]
            
            is_estimate = estimate_standard_is(dataset, behavior_policy, policy)
            results["standard_is"][policy_name]["estimates"].append(is_estimate)

            is_imputed_estimate = estimate_standard_is_with_imputation(dataset, policy, n_states, n_arms)
            results["standard_is_imputed"][policy_name]["estimates"].append(is_imputed_estimate)
            
            cstar_full_estimates = []
            for trial in range(n_annotation_trials):
                trial_seed = seed + trial * 10000
                trial_full_cf = generate_full_counterfactuals_with_consistent_seed(dataset, env, trial_seed)
                
                cstar_estimate = estimate_contextual_cstar_is(
                    trial_full_cf, policy, env.n_states, env.n_arms)
                cstar_full_estimates.append(cstar_estimate)
            
            results["cstar_is_full"][policy_name]["estimates"].append(np.mean(cstar_full_estimates))
            
            for budget in annotation_budgets:
                from contextual_bandit import generate_contextual_budget_limited_counterfactuals
                
                uniform_estimates = []
                for trial in range(n_annotation_trials):
                    trial_seed = seed + trial * 10000
                    trial_uniform_cf = generate_contextual_budget_limited_counterfactuals(
                        dataset, env, budget, trial_seed)
                    
                    uniform_estimate = estimate_contextual_cstar_is(
                        trial_uniform_cf, policy, env.n_states, env.n_arms)
                    uniform_estimates.append(uniform_estimate)
                
                if f"uniform_random_budget_{budget}" not in results:
                    results[f"uniform_random_budget_{budget}"] = {
                        policy_name: {"estimates": [], "rmse": 0, "std": 0, "mean": 0}
                        for policy_name in policy_names
                    }
                results[f"uniform_random_budget_{budget}"][policy_name]["estimates"].append(np.mean(uniform_estimates))
                
                for batch_size in batch_sizes:
                    batch_str = "single" if batch_size is None else str(batch_size)
                    
                    div_estimates = []
                    for trial in range(n_annotation_trials):
                        trial_seed = seed + trial * 10000
                        trial_div_cf = generate_divergence_prioritized_counterfactuals(
                            dataset, env, policy, behavior_policy, budget, batch_size, trial_seed)
                        
                        div_estimate = estimate_contextual_cstar_is(
                            trial_div_cf, policy, env.n_states, env.n_arms)
                        div_estimates.append(div_estimate)
                    
                    key = f"divergence_prioritized_budget_{budget}_batch_{batch_str}"
                    results[key][policy_name]["estimates"].append(np.mean(div_estimates))
                    
                    var_factual_estimates = []
                    for trial in range(n_annotation_trials):
                        trial_seed = seed + trial * 10000
                        trial_var_cf = generate_variance_prioritized_counterfactuals_factual_init(
                            dataset, env, budget, batch_size, trial_seed)
                        
                        var_estimate = estimate_contextual_cstar_is(
                            trial_var_cf, policy, env.n_states, env.n_arms)
                        var_factual_estimates.append(var_estimate)
                    
                    key = f"variance_prioritized_factual_init_budget_{budget}_batch_{batch_str}"
                    results[key][policy_name]["estimates"].append(np.mean(var_factual_estimates))
                    
                    var_uniform_estimates = []
                    for trial in range(n_annotation_trials):
                        trial_seed = seed + trial * 10000
                        trial_var_cf = generate_variance_prioritized_counterfactuals_uniform_init(
                            dataset, env, budget, batch_size, trial_seed)
                        
                        var_estimate = estimate_contextual_cstar_is(
                            trial_var_cf, policy, env.n_states, env.n_arms)
                        var_uniform_estimates.append(var_estimate)
                    
                    key = f"variance_prioritized_uniform_init_budget_{budget}_batch_{batch_str}"
                    results[key][policy_name]["estimates"].append(np.mean(var_uniform_estimates))
                    
                    hybrid_estimates = []
                    for trial in range(n_annotation_trials):
                        trial_seed = seed + trial * 10000
                        trial_hybrid_cf = generate_hybrid_divergence_variance_counterfactuals(
                            dataset, env, policy, behavior_policy, budget, batch_size, trial_seed)
                        
                        hybrid_estimate = estimate_contextual_cstar_is(
                            trial_hybrid_cf, policy, env.n_states, env.n_arms)
                        hybrid_estimates.append(hybrid_estimate)
                    
                    key = f"hybrid_divergence_variance_budget_{budget}_batch_{batch_str}"
                    results[key][policy_name]["estimates"].append(np.mean(hybrid_estimates))
    
    for approach in results:
        for policy_name in policy_names:
            estimates = results[approach][policy_name]["estimates"]
            true_value = true_values[policy_names.index(policy_name)]
            
            results[approach][policy_name]["mean"] = np.mean(estimates)
            results[approach][policy_name]["std"] = np.std(estimates)
            results[approach][policy_name]["rmse"] = np.sqrt(np.mean([(est - true_value)**2 for est in estimates]))
    
    return results, true_values

def plot_active_learning_results(results, true_values, policy_names, annotation_budgets, 
                                 batch_size="single", save_path=None):
    """
    Plot comparison of active learning approaches vs baselines
    """
    fig, axes = plt.subplots(1, len(policy_names), figsize=(15, 5))
    
    if len(policy_names) == 1:
        axes = [axes]
    
    for i, policy_name in enumerate(policy_names):
        uniform_rmse = []
        divergence_rmse = []
        variance_factual_rmse = []
        variance_uniform_rmse = []
        hybrid_rmse = []
        
        for budget in annotation_budgets:
            batch_str = batch_size
            
            uniform_key = f"uniform_random_budget_{budget}"
            div_key = f"divergence_prioritized_budget_{budget}_batch_{batch_str}"
            var_fact_key = f"variance_prioritized_factual_init_budget_{budget}_batch_{batch_str}"
            var_uni_key = f"variance_prioritized_uniform_init_budget_{budget}_batch_{batch_str}"
            hybrid_key = f"hybrid_divergence_variance_budget_{budget}_batch_{batch_str}"
            
            uniform_rmse.append(results[uniform_key][policy_name]["rmse"])
            divergence_rmse.append(results[div_key][policy_name]["rmse"])
            variance_factual_rmse.append(results[var_fact_key][policy_name]["rmse"])
            variance_uniform_rmse.append(results[var_uni_key][policy_name]["rmse"])
            hybrid_rmse.append(results[hybrid_key][policy_name]["rmse"])
        
        axes[i].plot(annotation_budgets, uniform_rmse, 'o-', linewidth=2, markersize=6, 
                    label='Uniform Random', color='gray')
        axes[i].plot(annotation_budgets, divergence_rmse, 's-', linewidth=2, markersize=6, 
                    label='Divergence Prioritized', color='red')
        axes[i].plot(annotation_budgets, variance_factual_rmse, '^-', linewidth=2, markersize=6, 
                    label='Variance Prioritized (Factual)', color='green')
        axes[i].plot(annotation_budgets, variance_uniform_rmse, 'v-', linewidth=2, markersize=6, 
                    label='Variance Prioritized (10% Uniform)', color='blue')
        axes[i].plot(annotation_budgets, hybrid_rmse, 'd-', linewidth=2, markersize=6, 
                    label='Hybrid (Div + Var)', color='purple')
        
        axes[i].axhline(y=results["standard_is"][policy_name]["rmse"],
                       color='orange', linestyle='--', linewidth=2, label='Standard IS')
        axes[i].axhline(y=results["standard_is_imputed"][policy_name]["rmse"],
                       color='brown', linestyle='--', linewidth=2, label='Standard IS + Imputation')
        axes[i].axhline(y=results["cstar_is_full"][policy_name]["rmse"],
                       color='black', linestyle=':', linewidth=2, label='C*-IS (Full Annotations)')
        
        axes[i].set_xlabel('Annotation Budget')
        axes[i].set_ylabel('RMSE')
        axes[i].set_title(f'{policy_name} Policy')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{save_path}'")
    
    return fig

def save_results_to_csv(results, true_values, policy_names, save_path):
    """
    Save detailed results to CSV file
    """
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
    df.to_csv(save_path, index=False)
    print(f"Detailed results saved as '{save_path}'")
    
    return df

if __name__ == "__main__":
    env = ContextualBandit(n_states=3, n_arms=5, seed=42)
    
    policies = [
        {0: [1, 0, 0, 0, 0], 1: [0, 0, 1, 0, 0], 2: [0, 0, 0, 0, 1]},
        {0: [0.2, 0.2, 0.2, 0.2, 0.2], 1: [0.2, 0.2, 0.2, 0.2, 0.2], 2: [0.2, 0.2, 0.2, 0.2, 0.2]},
        {0: [0, 1, 0, 0, 0], 1: [0, 0, 0, 1, 0], 2: [0.25, 0.25, 0.25, 0.25, 0]}
    ]
    
    policy_names = ["State-Optimal", "Uniform", "Suboptimal"]
    
    results, true_values = evaluate_active_approaches(
        env, policies, policy_names,
        n_datasets=50,
        samples_per_dataset=1000,
        annotation_budgets=[0, 500, 1000, 1500, 2000, 3000],
        batch_sizes=[None],
        n_annotation_trials=10
    )
    
    print("\n" + "="*80)
    print("ACTIVE LEARNING RESULTS")
    print("="*80)
    
    for i, policy_name in enumerate(policy_names):
        print(f"\n{policy_name} Policy (True Value: {true_values[i]:.3f})")
        print("-" * 80)
        
        print("BASELINES:")
        mean_est = results["standard_is"][policy_name]["mean"]
        rmse = results["standard_is"][policy_name]["rmse"]
        print(f"{'Standard IS':40s}: Mean={mean_est:.3f}, RMSE={rmse:.3f}")

        mean_est = results["standard_is_imputed"][policy_name]["mean"]
        rmse = results["standard_is_imputed"][policy_name]["rmse"]
        print(f"{'Standard IS + Imputation':40s}: Mean={mean_est:.3f}, RMSE={rmse:.3f}")
        
        mean_est = results["cstar_is_full"][policy_name]["mean"]
        rmse = results["cstar_is_full"][policy_name]["rmse"]
        print(f"{'C*-IS (full annotations)':40s}: Mean={mean_est:.3f}, RMSE={rmse:.3f}")
        
        print("\nACTIVE LEARNING (Budget = 2000):")
        budget = 2000
        batch_str = "single"
        
        key = f"uniform_random_budget_{budget}"
        mean_est = results[key][policy_name]["mean"]
        rmse = results[key][policy_name]["rmse"]
        print(f"{'Uniform Random':40s}: Mean={mean_est:.3f}, RMSE={rmse:.3f}")
        
        key = f"divergence_prioritized_budget_{budget}_batch_{batch_str}"
        mean_est = results[key][policy_name]["mean"]
        rmse = results[key][policy_name]["rmse"]
        print(f"{'Divergence Prioritized':40s}: Mean={mean_est:.3f}, RMSE={rmse:.3f}")
        
        key = f"variance_prioritized_factual_init_budget_{budget}_batch_{batch_str}"
        mean_est = results[key][policy_name]["mean"]
        rmse = results[key][policy_name]["rmse"]
        print(f"{'Variance Prioritized (Factual)':40s}: Mean={mean_est:.3f}, RMSE={rmse:.3f}")
        
        key = f"variance_prioritized_uniform_init_budget_{budget}_batch_{batch_str}"
        mean_est = results[key][policy_name]["mean"]
        rmse = results[key][policy_name]["rmse"]
        print(f"{'Variance Prioritized (10% Uniform)':40s}: Mean={mean_est:.3f}, RMSE={rmse:.3f}")
        
        key = f"hybrid_divergence_variance_budget_{budget}_batch_{batch_str}"
        mean_est = results[key][policy_name]["mean"]
        rmse = results[key][policy_name]["rmse"]
        print(f"{'Hybrid (Div + Var)':40s}: Mean={mean_est:.3f}, RMSE={rmse:.3f}")
    
    images_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'images')
    os.makedirs(images_dir, exist_ok=True)
    plot_path = os.path.join(images_dir, 'active_learning_results.png')
    
    fig = plot_active_learning_results(
        results, true_values, policy_names,
        annotation_budgets=[500, 1000, 1500, 2000, 3000],
        batch_size="single",
        save_path=plot_path
    )
    
    outputs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    csv_path = os.path.join(outputs_dir, 'active_learning_results.csv')
    
    df = save_results_to_csv(results, true_values, policy_names, csv_path)
    
    plt.show()