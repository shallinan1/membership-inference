import matplotlib.pyplot as plt
import numpy as np

# Example data structure
performance = {
    'Coverage': [0.65, 0.72, 0.78, 0.83, 0.88],
    'Creativity': [0.60, 0.67, 0.74, 0.80, 0.85],
    'LCS (word)': [0.62, 0.69, 0.75, 0.81, 0.86]
}
generation_lengths = [10, 25, 50, 100, 200]

# Plot
plt.figure(figsize=(4,4))

for metric, scores in performance.items():
    plt.plot(generation_lengths, scores, 'o--', label=metric)

plt.xlabel('Generation Length')
plt.ylabel('Performance')
plt.yticks(np.arange(0.5, 1.0, 0.1))
plt.title('Performance vs. Generation Length')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("code/plotting/plots/metrics.png", dpi=200, bbox_inches="tight")

"""
python3 -m code.plotting.plot_metrics
"""