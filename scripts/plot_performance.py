import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_performance(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run the C++ executable first.")
        return

    df = pd.read_csv(csv_file)

    # Filter out operations without CPU times (like memory transfers) for comparison chart
    compare_df = df[df['CPU_Time_ms'] > 0]

    # Plot 1: Execution Time Comparison
    plt.figure(figsize=(10, 6))
    x = range(len(compare_df['Operation']))
    width = 0.35

    plt.bar([i - width/2 for i in x], compare_df['CPU_Time_ms'], width, label='CPU (OpenCV)', color='skyblue')
    plt.bar([i + width/2 for i in x], compare_df['GPU_Time_ms'], width, label='GPU (CUDA)', color='orange')

    plt.xlabel('Image Processing Operation')
    plt.ylabel('Execution Time (ms)')
    plt.title('CPU vs GPU Execution Time Comparison')
    plt.xticks(x, compare_df['Operation'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/execution_time_comparison.png')
    print("Saved execution_time_comparison.png")

    # Plot 2: Speedup Factor
    plt.figure(figsize=(10, 6))
    plt.bar(compare_df['Operation'], compare_df['Speedup'], color='lightgreen')
    plt.axhline(y=1, color='r', linestyle='--', label='1x Speedup (Baseline)')
    
    plt.xlabel('Image Processing Operation')
    plt.ylabel('Speedup Factor (CPU_Time / GPU_Time)')
    plt.title('GPU Acceleration Speedup')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/speedup_factor.png')
    print("Saved speedup_factor.png")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    plot_performance("results/performance_metrics.csv")
