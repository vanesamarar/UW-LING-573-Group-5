# Author: Nina Koh
# Plots line charts for ROUGE scores across hyperparameter combinations

import pandas as pd
import matplotlib.pyplot as plt

def plot_rouge():
    """Plots ROUGE scores across hyperparameter combinations"""
    # Load data into a dataframe
    data = pd.read_csv("hyperparameter_tuning.csv")
    
    # Build a config label (e.g., "LR=2e-5, BS=8, E=3") for each row
    data["Config"] = data.apply(
        lambda row: f"LR={row['Learning Rate']}, BS={int(row['Training Batch Size'])}, E={int(row['Epoch'])}",
        axis=1
    )
    
    # Sort rows based on ROUGE-1 F1 score
    data = data.sort_values(by="Rouge1 F1", ascending=True).reset_index(drop=True)
    
    plt.figure(figsize=(12, 6))
    
    # Create x-axis labels (configs)
    x_axis = data["Config"]
    
    # Create y-axis labels (F1 scores) & plot three lines
    plt.plot(x_axis, data["Rouge1 F1"], label="ROUGE-1 F1", marker="o", color="darkseagreen")
    plt.plot(x_axis, data["Rouge2 F1"], label="ROUGE-2 F1", marker="o", color="teal")
    plt.plot(x_axis, data["Rougel F1"], label="ROUGE-L F1", marker="o", color="orchid")
    
    # Highlight the best performing configs
    best_idx = data["Rouge1 F1"].idxmax()
    plt.scatter(x_axis[best_idx], data["Rouge1 F1"][best_idx], color="darkseagreen", label="Best ROUGE-1", marker="*", s=200)
    
    best_idx = data["Rouge2 F1"].idxmax()
    plt.scatter(x_axis[best_idx], data["Rouge2 F1"][best_idx], color="teal", label="Best ROUGE-2", marker="*", s=200)
    
    best_idx = data["Rougel F1"].idxmax()
    plt.scatter(x_axis[best_idx], data["Rougel F1"][best_idx], color="orchid", label="Best ROUGE-L", marker="*", s=200)
    
    # Add title and labels
    plt.title("ROUGE Scores Across Hyperparameter Combinations")
    plt.xlabel("Hyperparameter Configurations")
    plt.ylabel("ROUGE F1 Scores")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.grid(True)
    
    plt.show() # Display the plot
    
if __name__ == "__main__":
    plot_rouge()