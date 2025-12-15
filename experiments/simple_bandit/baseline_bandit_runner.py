import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from baseline_bandit import (
    FiveArmedBandit,
    generate_dataset,
    generate_full_counterfactuals,
    generate_budget_limited_counterfactuals,
    estimate_full_cis,
    estimate_budget_limited_cis
)

# Get the absolute path to the experiments directory (parent of simple_bandit)
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(EXPERIMENTS_DIR, "images")
OUTPUTS_DIR = os.path.join(EXPERIMENTS_DIR, "outputs")



def evaluate_budget_limited_annotations(env, policies, policy_names, n_datasets=100, samples_per_dataset=1000, annotation_budgets=[100, 200, 500, 700]):
    """
    Evaluate policy estimation with varying annotation budgets
    
    Args:
        env (FiveArmedBandit): The bandit environment
        policies (list): List of policies to evaluate
        policy_names (list): Names for each policy
        n_datasets (int): Number of datasets to generate
        samples_per_dataset (int): Number of samples per dataset
        annotation_budgets (list): List of different annotation budgets to test
    
    Returns:
        dict: Results including RMSE and std for each policy and annotation budget
    """
    behavior_policy = [0.5, 0.2, 0.2, 0.05, 0.05]
    
    true_values = [env.get_true_value(policy) for policy in policies]
    
    results = {
        "full_annotations": {policy_name: {"estimates": [], "rmse": 0, "std": 0, "mean": 0} 
                            for policy_name in policy_names},
    }
    
    for budget in annotation_budgets:
        results[f"{budget}_annotations"] = {
            policy_name: {"estimates": [], "rmse": 0, "std": 0, "mean": 0} 
            for policy_name in policy_names
        }
    
    for dataset_idx in tqdm(range(n_datasets), desc="Generating datasets"):
        np.random.seed(1000 + dataset_idx)
        
        dataset = generate_dataset(env, behavior_policy, samples_per_dataset)
        
        # Calculate full budget
        full_budget = samples_per_dataset * (env.n_arms - 1)
        
        for p_idx, (policy, policy_name) in enumerate(zip(policies, policy_names)):
            
            # Evaluate all budgets including full budget using the same function
            all_budgets = annotation_budgets + [full_budget]
            
            for budget in all_budgets:
                # Average over multiple random annotation selections to reduce bias
                n_annotation_trials = 10
                budget_estimates = []
                for trial in range(n_annotation_trials):
                    np.random.seed(2000 + dataset_idx + budget + trial * 10000)

                    # Generate dataset with budget-limited counterfactuals
                    budget_cf_data = generate_budget_limited_counterfactuals(env, dataset, budget)

                    # Estimate policy value
                    budget_estimate = estimate_budget_limited_cis(budget_cf_data, behavior_policy, policy)
                    budget_estimates.append(budget_estimate)

                # Take average of estimates across annotation trials
                avg_budget_estimate = np.mean(budget_estimates)
                
                # Store in appropriate results bucket
                if budget == full_budget:
                    results["full_annotations"][policy_name]["estimates"].append(avg_budget_estimate)
                else:
                    results[f"{budget}_annotations"][policy_name]["estimates"].append(avg_budget_estimate)
    
    # Calculate summary statistics
    for policy_idx, policy_name in enumerate(policy_names):
        true_val = true_values[policy_idx]
        
        full_estimates = np.array(results["full_annotations"][policy_name]["estimates"])
        results["full_annotations"][policy_name]["rmse"] = np.sqrt(np.mean((full_estimates - true_val)**2))
        results["full_annotations"][policy_name]["std"] = np.std(full_estimates)
        results["full_annotations"][policy_name]["mean"] = np.mean(full_estimates)
        
        for budget in annotation_budgets:
            budget_estimates = np.array(results[f"{budget}_annotations"][policy_name]["estimates"])
            results[f"{budget}_annotations"][policy_name]["rmse"] = np.sqrt(np.mean((budget_estimates - true_val)**2))
            results[f"{budget}_annotations"][policy_name]["std"] = np.std(budget_estimates)
            results[f"{budget}_annotations"][policy_name]["mean"] = np.mean(budget_estimates)
    
    return results, true_values


def print_estimates_array(results, policy_names, annotation_budgets):
    """
    Print out the array of estimates for each policy and annotation budget
    
    Args:
        results (dict): Output from evaluate_budget_limited_annotations
        policy_names (list): Names of policies
        annotation_budgets (list): Annotation budgets that were evaluated
    """
    print("\n=== ESTIMATES ARRAYS ===")
    
    for policy_name in policy_names:
        print(f"\nPolicy: {policy_name}")
        
        print("\nFull Annotations Estimates:")
        full_estimates = results["full_annotations"][policy_name]["estimates"]
        for i, est in enumerate(full_estimates):
            if i % 10 == 0 and i > 0:
                print()  
            print(f"{est:.4f}", end=" ")
        print()  
   
        for budget in annotation_budgets:
            print(f"\n{budget} Annotations Budget Estimates:")
            budget_estimates = results[f"{budget}_annotations"][policy_name]["estimates"]
            for i, est in enumerate(budget_estimates):
                if i % 10 == 0 and i > 0:
                    print() 
                print(f"{est:.4f}", end=" ")
            print() 


def calculate_expected_value_rmse(results, true_values, policy_names, annotation_budgets):
    """
    Calculate RMSE between expected policy values and actual estimates
    
    Args:
        results (dict): Output from evaluate_budget_limited_annotations
        true_values (list): True values for each policy
        policy_names (list): Names of policies
        annotation_budgets (list): Annotation budgets that were evaluated
        
    Returns:
        dict: RMSE values for each policy and annotation budget
    """
    expected_rmse = {
        "full_annotations": {policy_name: 0 for policy_name in policy_names},
    }
    
    for budget in annotation_budgets:
        expected_rmse[f"{budget}_annotations"] = {
            policy_name: 0 for policy_name in policy_names
        }
    
    for policy_idx, policy_name in enumerate(policy_names):
        true_val = true_values[policy_idx]
        
        full_estimates = np.array(results["full_annotations"][policy_name]["estimates"])
        expected_rmse["full_annotations"][policy_name] = np.sqrt(np.mean((full_estimates - true_val)**2))
        
        for budget in annotation_budgets:
            budget_estimates = np.array(results[f"{budget}_annotations"][policy_name]["estimates"])
            expected_rmse[f"{budget}_annotations"][policy_name] = np.sqrt(np.mean((budget_estimates - true_val)**2))
    
    return expected_rmse


def visualize_annotation_results(results, true_values, policy_names, annotation_budgets):
    """
    Visualize the RMSE and std for different annotation budgets
    
    Args:
        results (dict): Output from evaluate_budget_limited_annotations
        true_values (list): True values for each policy
        policy_names (list): Names of policies
        annotation_budgets (list): Annotation budgets that were evaluated
    """
    # Calculate full annotation budget for reference
    n_samples = 1000
    n_arms = 5
    full_budget = n_samples * (n_arms - 1)  # Each sample has (n_arms - 1) possible counterfactuals
    
    # Create a figure to show relationship between budget and overall RMSE
    fig_budget_vs_rmse, ax_budget_vs_rmse = plt.subplots(figsize=(12, 8))
    
    # Calculate overall RMSE for each budget
    overall_rmse_by_budget = {}
    for budget in annotation_budgets:
        overall_rmse_by_budget[budget] = 0
        for policy_name in policy_names:
            overall_rmse_by_budget[budget] += results[f"{budget}_annotations"][policy_name]["rmse"]**2
        overall_rmse_by_budget[budget] = np.sqrt(overall_rmse_by_budget[budget] / len(policy_names))
    
    # Calculate full annotation RMSE
    overall_full_rmse = 0
    for policy_name in policy_names:
        overall_full_rmse += results["full_annotations"][policy_name]["rmse"]**2
    overall_full_rmse = np.sqrt(overall_full_rmse / len(policy_names))
    
    # Plot the relationship
    budgets = annotation_budgets + [full_budget]  # Add full budget
    rmses = [overall_rmse_by_budget[budget] for budget in annotation_budgets] + [overall_full_rmse]
    
    ax_budget_vs_rmse.plot(budgets, rmses, 'o-', linewidth=2, markersize=8)
    ax_budget_vs_rmse.set_title("Overall RMSE vs Annotation Budget", fontsize=16)
    ax_budget_vs_rmse.set_xlabel("Annotation Budget", fontsize=14)
    ax_budget_vs_rmse.set_ylabel("Overall RMSE", fontsize=14)
    ax_budget_vs_rmse.grid(True)
    
    # Add annotations showing the budget as a percentage of full budget
    for i, budget in enumerate(budgets):
        percentage = (budget / full_budget) * 100
        ax_budget_vs_rmse.annotate(
            f"{percentage:.1f}%", 
            (budget, rmses[i]),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )
    
    plt.tight_layout()
    plt.show()
    
    # Create multi-panel plot with one subplot per policy (matching contextual_baseline.py style)
    budgets = annotation_budgets + [full_budget]
    
    # Create subplots - one per policy
    fig, axes = plt.subplots(1, len(policy_names), figsize=(15, 5))
    
    # Handle case where there's only one policy
    if len(policy_names) == 1:
        axes = [axes]
    
    for i, policy_name in enumerate(policy_names):
        # Get RMSE values for this policy
        rmse_values = [results[f"{budget}_annotations"][policy_name]["rmse"] for budget in annotation_budgets]
        rmse_values.append(results["full_annotations"][policy_name]["rmse"])
        
        # Plot the main line
        axes[i].plot(budgets, rmse_values, 'o-', linewidth=2, markersize=6, label='C-IS')
        
        # Set labels and title
        axes[i].set_xlabel('Annotation Budget')
        axes[i].set_ylabel('RMSE')
        axes[i].set_title(f'{policy_name} Policy')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary tables
    print("\nSummary of Results:")
    print("=" * 110)
    
    print("RMSE Summary:")
    header = "Policy             |"
    for budget in budgets:
        header += f" {budget:<10} |"
    print(header)
    print("-" * (len(header) + 10))
    
    total_rmse_by_budget = {budget: 0 for budget in budgets}
    
    for policy_name in policy_names:
        row = f"{policy_name:18} |"
        for budget in annotation_budgets:
            rmse = results[f"{budget}_annotations"][policy_name]["rmse"]
            row += f" {rmse:.6f}    |"
            total_rmse_by_budget[budget] += rmse**2
        
        full_rmse = results["full_annotations"][policy_name]["rmse"]
        row += f" {full_rmse:.6f}    |"
        total_rmse_by_budget[full_budget] += full_rmse**2
        print(row)
    
    print("-" * (len(header) + 10))
    row = f"Overall RMSE       |"
    for budget in budgets:
        overall_rmse = np.sqrt(total_rmse_by_budget[budget] / len(policy_names))
        row += f" {overall_rmse:.6f}    |"
    print(row)
    
    print("\nStandard Deviation Summary:")
    header = "Policy             |"
    for budget in budgets:
        header += f" {budget:<10} |"
    print(header)
    print("-" * (len(header) + 10))
    
    avg_std_by_budget = {budget: 0 for budget in budgets}
    
    for policy_name in policy_names:
        row = f"{policy_name:18} |"
        for budget in annotation_budgets:
            std = results[f"{budget}_annotations"][policy_name]["std"]
            row += f" {std:.6f}    |"
            avg_std_by_budget[budget] += std
        
        full_std = results["full_annotations"][policy_name]["std"]
        row += f" {full_std:.6f}    |"
        avg_std_by_budget[full_budget] += full_std
        print(row)
    
    print("-" * (len(header) + 10))
    row = f"Average StdDev     |"
    for budget in budgets:
        avg_std = avg_std_by_budget[budget] / len(policy_names)
        row += f" {avg_std:.6f}    |"
    print(row)

    print("\nMean Estimates Summary:")
    header = "Policy             |"
    for budget in budgets:
        header += f" {budget:<10} |"
    print(header)
    print("-" * (len(header) + 10))
    
    total_bias_by_budget = {budget: 0 for budget in budgets}
    
    for policy_idx, policy_name in enumerate(policy_names):
        true_val = true_values[policy_idx]
        row = f"{policy_name:18} |"
        
        for budget in annotation_budgets:
            mean = results[f"{budget}_annotations"][policy_name]["mean"]
            row += f" {mean:.6f}    |"
            total_bias_by_budget[budget] += abs(mean - true_val)
        
        full_mean = results["full_annotations"][policy_name]["mean"]
        row += f" {full_mean:.6f}    |"
        total_bias_by_budget[full_budget] += abs(full_mean - true_val)
        print(row)
    
    print("-" * (len(header) + 10))
    row = f"Average Abs Bias   |"
    for budget in budgets:
        avg_bias = total_bias_by_budget[budget] / len(policy_names)
        row += f" {avg_bias:.6f}    |"
    print(row)
    
    print("=" * 110)
    print("\nTrue Values:")
    for policy_idx, policy_name in enumerate(policy_names):
        print(f"{policy_name:18}: {true_values[policy_idx]:.6f}")


def visualize_expected_value_rmse(expected_rmse, policy_names, annotation_budgets):
    """
    Visualize the RMSE between expected policy values and actual estimates
    
    Args:
        expected_rmse (dict): Output from calculate_expected_value_rmse
        policy_names (list): Names of policies
        annotation_budgets (list): Annotation budgets that were evaluated
    """
    # Calculate full annotation budget for reference
    n_samples = 1000
    n_arms = 5
    full_budget = n_samples * (n_arms - 1)
    
    budgets = annotation_budgets + [full_budget]
    x_labels = [str(budget) for budget in budgets]
    
    fig_expected, ax_expected = plt.subplots(figsize=(10, 6))
    
    for policy_name in policy_names:
        rmse_values = [expected_rmse[f"{budget}_annotations"][policy_name] for budget in annotation_budgets]
        rmse_values.append(expected_rmse["full_annotations"][policy_name])
        
        ax_expected.plot(budgets, rmse_values, marker='o', label=policy_name)
    
    ax_expected.set_title("RMSE vs Annotation Budget", fontsize=16)
    ax_expected.set_xlabel("Annotation Budget", fontsize=14)
    ax_expected.set_ylabel("Expected Value RMSE", fontsize=14)
    ax_expected.set_xticks(budgets)
    ax_expected.set_xticklabels(x_labels)
    ax_expected.legend()
    ax_expected.grid(True)
    
    plt.tight_layout()
    fig_expected.savefig(os.path.join(IMAGES_DIR, "expected_value_rmse_vs_budget.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(IMAGES_DIR, 'expected_value_rmse_vs_budget.png')}")
    plt.close()
    
    print("\nExpected Value RMSE Summary:")
    header = "Policy             |"
    for budget in budgets:
        header += f" {budget:<10} |"
    print(header)
    print("-" * (len(header) + 10))
    
    total_expected_rmse_by_budget = {budget: 0 for budget in budgets}
    
    for policy_name in policy_names:
        row = f"{policy_name:18} |"
        for budget in annotation_budgets:
            rmse = expected_rmse[f"{budget}_annotations"][policy_name]
            row += f" {rmse:.6f}    |"
            total_expected_rmse_by_budget[budget] += rmse**2
        
        full_rmse = expected_rmse["full_annotations"][policy_name]
        row += f" {full_rmse:.6f}    |"
        total_expected_rmse_by_budget[full_budget] += full_rmse**2
        print(row)
    
    print("-" * (len(header) + 10))
    row = f"Overall Expected RMSE |"
    for budget in budgets:
        overall_rmse = np.sqrt(total_expected_rmse_by_budget[budget] / len(policy_names))
        row += f" {overall_rmse:.6f}    |"
    print(row)


def save_rmse_to_csv(results, policy_names, annotation_budgets, filename="rmse_results.csv"):
    """
    Save RMSE values for each policy at each annotation budget to a CSV file
    
    Args:
        results (dict): Output from evaluate_budget_limited_annotations
        policy_names (list): Names of policies
        annotation_budgets (list): Annotation budgets that were evaluated
        filename (str): Name of the output CSV file
    """
    import csv
    
    filepath = os.path.join(OUTPUTS_DIR, filename)
    print(f"\n=== Saving RMSE values to {filepath} ===")
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row with budget values in correct order
        header = ['Policy'] + [str(budget) for budget in annotation_budgets] + ['Full (4000)']
        writer.writerow(header)
        
        # Write RMSE for each policy
        for policy_name in policy_names:
            row = [policy_name]
            for budget in annotation_budgets:
                budget_rmse = results[f"{budget}_annotations"][policy_name]["rmse"]
                row.append(f"{budget_rmse:.6f}")
            full_rmse = results["full_annotations"][policy_name]["rmse"]
            row.append(f"{full_rmse:.6f}")
            writer.writerow(row)
    
    print(f"RMSE values successfully saved to {filepath}")
    
    # Print a preview of the data
    print("\nRMSE Values Preview (first few policies and budgets):")
    preview_policies = policy_names[:3] if len(policy_names) > 3 else policy_names
    preview_budgets = annotation_budgets[:3] if len(annotation_budgets) > 3 else annotation_budgets
    
    header = f"{'Policy':<15} |"
    for budget in preview_budgets:
        header += f" {budget:<10} |"
    header += f" {'Full (4000)':<10} |"
    print(header)
    print("-" * (len(header) + 10))
    
    for policy_name in preview_policies:
        line = f"{policy_name:<15} |"
        for budget in preview_budgets:
            budget_rmse = results[f"{budget}_annotations"][policy_name]["rmse"]
            line += f" {budget_rmse:<10.6f} |"
        full_rmse = results["full_annotations"][policy_name]["rmse"]
        line += f" {full_rmse:<10.6f} |"
        print(line)
    
    print(f"... (See {filepath} for complete data)")


def run_evaluation():
    """
    Run the complete evaluation
    """
    print("Starting budget-limited annotation evaluation...")
    print(f"Images will be saved to: {os.path.abspath(IMAGES_DIR)}")
    print(f"CSV files will be saved to: {os.path.abspath(OUTPUTS_DIR)}\n")
    
    env = FiveArmedBandit(seed=42)
    
    optimal = [0, 0, 0, 0, 1]
    uniform = [0.2, 0.2, 0.2, 0.2, 0.2]
    near_optimal = [0.05, 0.05, 0.05, 0.05, 0.8]
    near_worst = [0.8, 0.1, 0.05, 0.025, 0.025]
    worst = [1, 0, 0, 0, 0]

    policies = [optimal, uniform, near_optimal, near_worst, worst]
    policy_names = ["Optimal", "Uniform", "Near-Optimal", "Near-Worst", "Worst"]
        
    annotation_budgets = []
    sum = 0
    while sum <= 3500:
        annotation_budgets.append(sum)
        sum += 500
    
    results, true_values = evaluate_budget_limited_annotations(
        env=env,
        policies=policies,
        policy_names=policy_names,
        n_datasets=100,
        samples_per_dataset=1000,
        annotation_budgets=annotation_budgets
    )

    print_estimates_array(results, policy_names, annotation_budgets)
    
    save_rmse_to_csv(results, policy_names, annotation_budgets, filename="rmse_by_budget.csv")
    
    expected_rmse = calculate_expected_value_rmse(results, true_values, policy_names, annotation_budgets)
    visualize_expected_value_rmse(expected_rmse, policy_names, annotation_budgets)
    
    visualize_annotation_results(results, true_values, policy_names, annotation_budgets)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  Images: {os.path.abspath(IMAGES_DIR)}/")
    print(f"    - overall_rmse_vs_budget.png")
    print(f"    - rmse_vs_budget.png")
    print(f"    - std_vs_budget.png")
    print(f"    - expected_value_rmse_vs_budget.png")
    print(f"  Data: {os.path.abspath(OUTPUTS_DIR)}/")
    print(f"    - rmse_by_budget.csv")
    print("="*80)


if __name__ == "__main__":
    run_evaluation()

