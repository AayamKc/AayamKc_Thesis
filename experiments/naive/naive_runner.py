import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '2_contextual_bandot'))

from contextual_bandit import ContextualBandit, generate_contextual_dataset, impute_missing_contextual_counterfactuals
from contextual_bandit_runner import estimate_contextual_cstar_is, estimate_contextual_cis

from naive import (
    generate_uniform_random_counterfactuals,
    generate_greedy_pie_only_counterfactuals,
    generate_greedy_support_based_counterfactuals,
    generate_round_robin_counterfactuals,
    generate_high_reward_first_counterfactuals
)


def evaluate_naive_approaches(env, policies, policy_names, behavior_policy, 
                             n_datasets=50, samples_per_dataset=1000, 
                             annotation_budgets=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]):
    """
    Evaluate all naive annotation solicitation approaches against contextual bandit
    
    Args:
        env: ContextualBandit environment
        policies: List of evaluation policies
        policy_names: Names for each policy
        behavior_policy: Dict mapping state -> list of action probabilities
        n_datasets: Number of datasets to generate
        samples_per_dataset: Samples per dataset  
        annotation_budgets: List of annotation budget sizes
        
    Returns:
        dict: Results with RMSE, std, mean for each approach and policy
    """
    n_states = env.n_states
    n_arms = env.n_arms
    true_values = [env.get_true_value(policy) for policy in policies]
    
    approaches = [
        "uniform_random",
        "greedy_pie_only", 
        "greedy_support_based",
        "round_robin",
        "high_reward_first"
    ]
    
    results = {}
    for approach in approaches:
        results[approach] = {}
        for budget in annotation_budgets:
            results[approach][f"budget_{budget}"] = {
                policy_name: {"estimates": [], "rmse": 0, "std": 0, "mean": 0}
                for policy_name in policy_names
            }
    
    print("Running naive approach evaluation...")
    
    for dataset_idx in tqdm(range(n_datasets), desc="Generating datasets"):
        seed = dataset_idx
        dataset = generate_contextual_dataset(env, behavior_policy, samples_per_dataset, seed)
        
        for policy_idx, (policy, policy_name) in enumerate(zip(policies, policy_names)):
            
            for budget in annotation_budgets:
                n_annotation_trials = 10
                
                for approach in approaches:
                    approach_estimates = []
                    
                    for trial in range(n_annotation_trials):
                        trial_seed = seed + trial * 10000
                        
                        if approach == "uniform_random":
                            cf_data = generate_uniform_random_counterfactuals(
                                dataset, env, budget, trial_seed)
                        elif approach == "greedy_pie_only":
                            cf_data = generate_greedy_pie_only_counterfactuals(
                                dataset, env, policy, budget, trial_seed)
                        elif approach == "greedy_support_based":
                            cf_data = generate_greedy_support_based_counterfactuals(
                                dataset, env, behavior_policy, budget, trial_seed)
                        elif approach == "round_robin":
                            cf_data = generate_round_robin_counterfactuals(
                                dataset, env, budget, trial_seed)
                        elif approach == "high_reward_first":
                            cf_data = generate_high_reward_first_counterfactuals(
                                dataset, env, budget, trial_seed)
                        
                        estimate = estimate_contextual_cstar_is(cf_data, policy, n_states, n_arms)
                        approach_estimates.append(estimate)
                    
                    avg_estimate = np.mean(approach_estimates)
                    results[approach][f"budget_{budget}"][policy_name]["estimates"].append(avg_estimate)
    
    for approach in approaches:
        for budget in annotation_budgets:
            for policy_idx, policy_name in enumerate(policy_names):
                estimates = results[approach][f"budget_{budget}"][policy_name]["estimates"]
                true_value = true_values[policy_idx]
                
                results[approach][f"budget_{budget}"][policy_name]["mean"] = np.mean(estimates)
                results[approach][f"budget_{budget}"][policy_name]["std"] = np.std(estimates)
                results[approach][f"budget_{budget}"][policy_name]["rmse"] = np.sqrt(
                    np.mean([(est - true_value)**2 for est in estimates]))
    
    return results, true_values


def visualize_comparison_results(results, true_values, policy_names, annotation_budgets):
    """
    Create comparison plots showing all approaches across different budgets
    """
    approaches = list(results.keys())
    approach_labels = {
        "uniform_random": "Uniform Random",
        "greedy_pie_only": "Greedy πe-only", 
        "greedy_support_based": "Greedy Support-based",
        "round_robin": "Round-robin",
        "high_reward_first": "High-reward-first"
    }
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    fig, axes = plt.subplots(1, len(policy_names), figsize=(15, 5))
    
    if len(policy_names) == 1:
        axes = [axes]
    
    for policy_idx, policy_name in enumerate(policy_names):
        for approach_idx, approach in enumerate(approaches):
            rmse_values = []
            for budget in annotation_budgets:
                rmse = results[approach][f"budget_{budget}"][policy_name]["rmse"]
                rmse_values.append(rmse)
            
            axes[policy_idx].plot(annotation_budgets, rmse_values, 
                                color=colors[approach_idx], marker=markers[approach_idx],
                                linewidth=2, markersize=6, 
                                label=approach_labels[approach])
        
        axes[policy_idx].set_xlabel('Annotation Budget')
        axes[policy_idx].set_ylabel('RMSE')
        axes[policy_idx].set_title(f'{policy_name} Policy')
        axes[policy_idx].grid(True, alpha=0.3)
        axes[policy_idx].legend()
    
    plt.tight_layout()
    
    # Save plot
    images_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'images')
    os.makedirs(images_dir, exist_ok=True)
    plot_path = os.path.join(images_dir, 'naive_approaches_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved as '{plot_path}'")
    plt.show()
    
    visualize_average_rmse(results, policy_names, annotation_budgets)


def visualize_average_rmse(results, policy_names, annotation_budgets):
    """
    Create a plot showing average RMSE across all policies for each approach
    """
    approaches = list(results.keys())
    approach_labels = {
        "uniform_random": "Uniform Random",
        "greedy_pie_only": "Greedy πe-only", 
        "greedy_support_based": "Greedy Support-based",
        "round_robin": "Round-robin",
        "high_reward_first": "High-reward-first"
    }
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for approach_idx, approach in enumerate(approaches):
        avg_rmse_values = []
        for budget in annotation_budgets:
            total_rmse_squared = 0
            for policy_name in policy_names:
                rmse = results[approach][f"budget_{budget}"][policy_name]["rmse"]
                total_rmse_squared += rmse**2
            avg_rmse = np.sqrt(total_rmse_squared / len(policy_names))
            avg_rmse_values.append(avg_rmse)
        
        ax.plot(annotation_budgets, avg_rmse_values, 
                color=colors[approach_idx], marker=markers[approach_idx],
                linewidth=3, markersize=8, 
                label=approach_labels[approach])
    
    ax.set_xlabel('Annotation Budget', fontsize=14)
    ax.set_ylabel('Average RMSE Across All Policies', fontsize=14)
    ax.set_title('Overall Performance Comparison: Average RMSE vs Annotation Budget', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    # Save plot
    images_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'images')
    os.makedirs(images_dir, exist_ok=True)
    avg_plot_path = os.path.join(images_dir, 'naive_approaches_average_rmse.png')
    plt.savefig(avg_plot_path, dpi=300, bbox_inches='tight')
    print(f"Average RMSE plot saved as '{avg_plot_path}'")
    plt.show()
    
    print("\n" + "="*80)
    print("AVERAGE RMSE ACROSS ALL POLICIES")
    print("="*80)
    
    header = f"{'Approach':<25}"
    for budget in annotation_budgets:
        header += f"Budget {budget:>6} "
    print(header)
    print("-" * len(header))
    
    for approach in approaches:
        row = f"{approach_labels[approach]:<25}"
        for budget in annotation_budgets:
            total_rmse_squared = 0
            for policy_name in policy_names:
                rmse = results[approach][f"budget_{budget}"][policy_name]["rmse"]
                total_rmse_squared += rmse**2
            avg_rmse = np.sqrt(total_rmse_squared / len(policy_names))
            row += f"{avg_rmse:>12.4f} "
        print(row)
    print("="*80)


def print_summary_tables(results, true_values, policy_names, annotation_budgets):
    """
    Print detailed summary tables comparing all approaches
    """
    approaches = list(results.keys())
    approach_labels = {
        "uniform_random": "Uniform Random",
        "greedy_pie_only": "Greedy πe-only", 
        "greedy_support_based": "Greedy Support-based",
        "round_robin": "Round-robin",
        "high_reward_first": "High-reward-first"
    }
    
    print("\n" + "="*120)
    print("NAIVE APPROACHES COMPARISON RESULTS")
    print("="*120)
    
    for policy_idx, policy_name in enumerate(policy_names):
        print(f"\n{policy_name} Policy (True Value: {true_values[policy_idx]:.4f})")
        print("-" * 120)
        
        header = f"{'Approach':<25}"
        for budget in annotation_budgets:
            header += f"Budget {budget:>6} "
        print(header)
        print("-" * 120)
        
        for approach in approaches:
            row = f"{approach_labels[approach]:<25}"
            for budget in annotation_budgets:
                rmse = results[approach][f"budget_{budget}"][policy_name]["rmse"]
                row += f"{rmse:>12.4f} "
            print(row)
    
    print("\n" + "="*120)
    print("OVERALL RMSE COMPARISON (Average across all policies)")
    print("="*120)
    
    header = f"{'Approach':<25}"
    for budget in annotation_budgets:
        header += f"Budget {budget:>6} "
    print(header)
    print("-" * 120)
    
    for approach in approaches:
        row = f"{approach_labels[approach]:<25}"
        for budget in annotation_budgets:
            overall_rmse = 0
            for policy_name in policy_names:
                rmse = results[approach][f"budget_{budget}"][policy_name]["rmse"]
                overall_rmse += rmse**2
            overall_rmse = np.sqrt(overall_rmse / len(policy_names))
            row += f"{overall_rmse:>12.4f} "
        print(row)


def save_results_to_csv(results, true_values, policy_names, annotation_budgets):
    """
    Save detailed results to CSV for further analysis
    """
    approaches = list(results.keys())
    
    detailed_results = []
    for approach in approaches:
        for budget in annotation_budgets:
            for policy_name in policy_names:
                policy_idx = policy_names.index(policy_name)
                true_value = true_values[policy_idx]
                
                result_data = results[approach][f"budget_{budget}"][policy_name]
                detailed_results.append({
                    'approach': approach,
                    'budget': budget,
                    'policy': policy_name,
                    'true_value': true_value,
                    'mean_estimate': result_data["mean"],
                    'rmse': result_data["rmse"],
                    'std': result_data["std"],
                    'bias': abs(result_data["mean"] - true_value)
                })
    
    df = pd.DataFrame(detailed_results)
    
    outputs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    csv_path = os.path.join(outputs_dir, 'naive_approaches_detailed_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved as '{csv_path}'")
    
    summary_results = []
    for approach in approaches:
        for budget in annotation_budgets:
            overall_rmse = 0
            overall_bias = 0
            overall_std = 0
            
            for policy_name in policy_names:
                policy_idx = policy_names.index(policy_name)
                true_value = true_values[policy_idx]
                
                result_data = results[approach][f"budget_{budget}"][policy_name]
                overall_rmse += result_data["rmse"]**2
                overall_bias += abs(result_data["mean"] - true_value)
                overall_std += result_data["std"]
            
            summary_results.append({
                'approach': approach,
                'budget': budget,
                'overall_rmse': np.sqrt(overall_rmse / len(policy_names)),
                'overall_bias': overall_bias / len(policy_names),
                'overall_std': overall_std / len(policy_names)
            })
    
    summary_df = pd.DataFrame(summary_results)
    summary_csv_path = os.path.join(outputs_dir, 'naive_approaches_summary_results.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary results saved as '{summary_csv_path}'")


def run_naive_evaluation():
    """
    Main function to run the complete naive approaches evaluation
    """
    print("="*80)
    print("NAIVE ANNOTATION SOLICITATION APPROACHES EVALUATION")
    print("="*80)
    print("Comparing naive approaches to contextual bandit baselines")
    print("Environment: Contextual bandit with 3 states, 5 arms")
    print("="*80)
    
    env = ContextualBandit(n_states=3, n_arms=5, seed=42)
    
    behavior_policy = {
        0: [0.4, 0.3, 0.1, 0.1, 0.1],
        1: [0.1, 0.1, 0.4, 0.3, 0.1],
        2: [0.2, 0.2, 0.2, 0.2, 0.2]   
    }
    
    policies = [
        {0: [1, 0, 0, 0, 0], 1: [0, 0, 1, 0, 0], 2: [0, 0, 0, 0, 1]},  # State-optimal
        {0: [0.2, 0.2, 0.2, 0.2, 0.2], 1: [0.2, 0.2, 0.2, 0.2, 0.2], 2: [0.2, 0.2, 0.2, 0.2, 0.2]},  # Uniform
        {0: [0, 1, 0, 0, 0], 1: [0, 0, 0, 1, 0], 2: [0.25, 0.25, 0.25, 0.25, 0]}  # Suboptimal
    ]
    policy_names = ["State-Optimal", "Uniform", "Suboptimal"]
    
    annotation_budgets = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    
    results, true_values = evaluate_naive_approaches(
        env=env,
        policies=policies, 
        policy_names=policy_names,
        behavior_policy=behavior_policy,
        n_datasets=50,  # Same as contextual bandit
        samples_per_dataset=1000,
        annotation_budgets=annotation_budgets
    )
    
    print_summary_tables(results, true_values, policy_names, annotation_budgets)
    visualize_comparison_results(results, true_values, policy_names, annotation_budgets)
    save_results_to_csv(results, true_values, policy_names, annotation_budgets)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"Results saved to outputs/ directory")
    print(f"Plots saved to images/ directory")
    print("="*80)


if __name__ == "__main__":
    run_naive_evaluation()